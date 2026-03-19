import os
import numpy as np
from tqdm import tqdm
from glob import glob
import json
import torch
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import src.utils.data_utils as data_utils
from src.modules.data_modules import load_dataloaders
from train_vertex_model import load_v_models
from train_face_model import load_f_models
import torch.multiprocessing as mp
from queue import Empty
import traceback
from pathlib import Path

def process_data(predv, pc_vertex_batch, device):
    pred_vertices = torch.from_numpy(predv['vertices'][:predv['num_vertices']]).float().to(device)
    pred_vertices = data_utils.quantize_verts(pred_vertices)
    pred_vertices, inv = torch.unique(pred_vertices, dim=0, return_inverse=True)
    sort_inds = data_utils.torch_lexsort(pred_vertices.T)
    pred_vertices = pred_vertices[sort_inds]
    pred_vertices = pred_vertices.to(torch.int32)
    pred_vertices = data_utils.dequantize_verts(pred_vertices)

    face_batch = {}
    face_batch["vertices"] = pred_vertices.unsqueeze(0)
    face_batch["vertices_mask"] = torch.ones_like(pred_vertices[..., 0], dtype=torch.float32).unsqueeze(0)
    face_batch["files_list"] = [item.replace('/xyz_n/', '/meshes/').replace('.xyz', '.obj') for item in pc_vertex_batch['filenames']]

    return face_batch

def sample_vertices(vertex_model, vertex_batch):
    with torch.no_grad():
        vertex_samples = vertex_model.sample_mask(context = vertex_batch, num_samples = vertex_batch["vertices_flat"].shape[0],
                                            max_sample_length = 100, top_p = 0.9, recenter_verts = False, only_return_complete = False)
    out_dict = {}
    out_dict["vertices"] = vertex_samples["vertices"][0].cpu().numpy()
    out_dict["num_vertices"] = vertex_samples["num_vertices"][0].cpu().numpy()
    return out_dict

def sample_faces(face_model, face_batch, top_p=0.9, return_mesh=False):
    with torch.no_grad():
        face_samples = face_model.sample_mask(context = face_batch, max_sample_length = 500, top_p = top_p, only_return_complete = False)
    curr_faces = face_samples["faces"][0]
    num_face_indices = face_samples['num_face_indices'][0]
    pred_faces = data_utils.unflatten_faces(curr_faces[:num_face_indices].detach().cpu().numpy())

    return pred_faces

def v_have_stop_token(vs):
    if len(vs) < 100:
        return True
    else:
        return False
    
def f_have_stop_token(fs):
    def compute_len_fs(fs):
        return len([item for sublist in fs for item in sublist])+len(fs)+1
    len_f = compute_len_fs(fs)
    if len(fs) < 500:
        return True
    else:
        return False

def is_floor_covering_pointcloudxy(vs, pts, info, coverage_rate_thres=0.7):
    vs_floor_inds = np.where(vs[:,-1]<vs[:,-1].min()+0.5/info['scale'])[0]
    points1 = vs[vs_floor_inds][:,:2]
    points2 = pts[:,:2]

    if len(points1) < 3:
        return False, None

    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    poly1 = Polygon(points1[hull1.vertices])
    poly2 = Polygon(points2[hull2.vertices])

    intersection = poly1.intersection(poly2)
    coverage_rate = intersection.area / poly2.area

    return coverage_rate > coverage_rate_thres, vs_floor_inds

def are_missing_floor_vertices(vs, vs_floor_inds):
    vs_floor = vs[vs_floor_inds][:,:2]
    hull = ConvexHull(vs_floor)
    hull_points = vs_floor[hull.vertices]

    angles = []
    num_points = len(hull_points)
    for i in range(num_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % num_points]
        p3 = hull_points[(i + 2) % num_points]
        angle = data_utils.calculate_angle(p1, p2, p3)
        angles.append(angle)

    return any(angle < 60 for angle in angles)

def are_missing_floor_faces(vs, fs, vs_floor_inds):
    floor_f_bool_list = [all(element in list(vs_floor_inds) for element in f) for f in fs]
    
    if np.sum(floor_f_bool_list) == 0:
        return True
    else:
        result_floor_fs = [fs[_] for _, mask_value in enumerate(floor_f_bool_list) if mask_value]
        valid_poly_bools = []
        for one_result_floor_fs in result_floor_fs:
            valid_poly_bools.append(Polygon(vs[:,:2][one_result_floor_fs]).is_valid)
        return not(all(valid_poly_bools))


def worker(gpu_id, work_queue, model_base_dir, test_data_dir, CITY, data_info, stats, results_dir):
    """Each worker owns one GPU and pulls items from a shared queue."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # Each process loads its own model copies onto its own GPU
    checkpoint_v_pth = os.path.join(model_base_dir, 'vertex_model', 'checkpoint_v.pth')
    checkpoint_v = torch.load(checkpoint_v_pth, map_location=device)
    pc_vertex_model = load_v_models(device=device)
    pc_vertex_model.to(device)
    pc_vertex_model.load_state_dict(checkpoint_v['state_dict'])

    checkpoint_f_pth = os.path.join(model_base_dir, 'face_model', 'checkpoint_f.pth')
    checkpoint_f = torch.load(checkpoint_f_pth, map_location=device)
    face_model = load_f_models(device=device)
    face_model.to(device)
    face_model.load_state_dict(checkpoint_f['state_dict'])

    pc_vertex_model.eval()
    face_model.eval()
    
    processed = 0
    failed = 0
    
    while True:
        try:
            j, pc_vertex_batch = work_queue.get(timeout=1)
        except Empty:
            break  # No more work

        # Move batch to this worker's GPU
        for k in pc_vertex_batch:
            if k != 'filenames':
                pc_vertex_batch[k] = pc_vertex_batch[k].to(device)

        one_data_info = data_info[
            os.path.split(pc_vertex_batch['filenames'][0])[-1].split('.')[0]
        ]
        try: 
            run_v_id = 0
            while run_v_id < 10:
                predv = sample_vertices(pc_vertex_model, pc_vertex_batch)
                if not v_have_stop_token(predv['vertices']):
                    run_v_id += 1
                    continue
                run_f_id = 0
                face_batch = process_data(predv, pc_vertex_batch, device)
                while run_f_id < 10:
                    # print('run {} iterations'.format(run_v_id*10+run_f_id))
                    predf = sample_faces(face_model, face_batch, return_mesh=True)
                    if not f_have_stop_token(predf):
                        run_f_id += 1
                        continue
                    if_floor_cover, vs_floor_inds = is_floor_covering_pointcloudxy(predv['vertices'], data_utils.dequantize_verts(pc_vertex_batch['pc_coords'][:,1:], 8).detach().cpu().numpy(), one_data_info)
                    if vs_floor_inds is None:
                        break
                    if not if_floor_cover:
                        if are_missing_floor_vertices(predv['vertices'], vs_floor_inds):
                            break
                        elif are_missing_floor_faces(predv['vertices'], predf, vs_floor_inds):
                            run_f_id += 1
                            continue

                    out_file = os.path.join(results_dir, os.path.split(face_batch['files_list'][0])[-1])
                    data_utils.process_and_save_mesh(vertices=face_batch['vertices'][0].detach().cpu().numpy(), faces=predf, file_path=out_file, precess_dup=True)
                    run_v_id = 10  
                    break

                run_v_id += 1
            processed += 1
        except Exception as err:
            print(Exception, err)
            print(traceback.format_exc())
            print(f"failed for {pc_vertex_batch['filenames']}")
            failed += 1
            torch.cuda.synchronize(device)
            continue
    print(f"[GPU {gpu_id}] Done — processed: {processed}, failed: {failed}")
    stats[gpu_id] = {'processed': processed, 'failed': failed}
    return processed, failed

def main(CITY,
    device,
    model_base_dir,
    test_data_dir,
    results_dir):

    mp.set_start_method('spawn', force=True)  # Required for CUDA
    manager = mp.Manager()
    stats = manager.dict()
    # Load dataloader & metadata in the main process
    all_pointcloud_files = sorted(glob(os.path.join(test_data_dir, 'xyz_n', '*.xyz')))
    pc_dataloader, pc_info_file = load_dataloaders(
        test_data_dir, all_pointcloud_files,
        batch_size=1, preprocess=True, data_split='test', CITY=CITY, stage=1,results_dir=results_dir
    )
    with open(pc_info_file) as f:
        data_info = json.load(f)

    # Fill a shared queue with all work items (tensors moved to CPU)
    work_queue = mp.Queue()
    for j, pc_vertex_batch in enumerate(pc_dataloader):
        # Ensure everything is on CPU before enqueueing
        for k in pc_vertex_batch:
            if k != 'filenames':
                pc_vertex_batch[k] = pc_vertex_batch[k].cpu()
        work_queue.put((j, pc_vertex_batch))
    total_items = work_queue.qsize()
    print(f"\nGoing to start. Total items queued: {total_items}")
    # Launch 5 workers, one per GPU
    num_gpus = 5
    processes = []
    for gpu_id in range(2,num_gpus+2):
        p = mp.Process(
            target=worker,
            args=(gpu_id, work_queue, model_base_dir, test_data_dir, CITY, data_info, stats, results_dir),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_processed = sum(s['processed'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())
    print(f"Total processed: {total_processed}, failed: {total_failed}")
    print(f"\nAll workers finished. Total items queued: {total_items}")

    

if __name__ == '__main__':
    CITY = 'Zuerich' #'VAIHINGEN' #
    device = 'cuda'
    model_base_dir = './saved_model'
    test_data_dir = "/home/jovyan/repos/review/data/Zuerich/" #"/home/jovyan/repos/review/data/Vaihingen/out_old/test" #'datasets/{}/testset'.format(CITY)
    results_dir = os.path.join('results', CITY)
    main(CITY, device, model_base_dir, test_data_dir, results_dir)
