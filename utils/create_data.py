import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from asyncio import Event
from typing import Tuple

import ray
from ray.actor import ActorHandle
from tqdm import tqdm

from magtense import magtense


def generate_exp(idx, res, datapath, t_start, check, ext,
                 test, lab_setup, orig, shared=False, pba=None):
    """ Generate 3-D magnetic fields of experimental setup
    Parameters:
        idx: Internal iterator. A new random generator is initiated.
        res: Resolution of magnetic field.
        datapath: Indicates where to store the data sample.
        t_start: Starting time for progress bar.
        check: Boolean for possible visual output.
        ext: Boolean if a layer above and below should be generated.
        test: Creating data with different seeds for random number generator.
        lab_setup: Generate data that is similar to our lab setup.
        orig: Generate identical magnetic field as the measured one.
        shared: True if multiprocessing is in use.
        pba: Ray actor for progress bar.
    """
    # Omit already created files for restart
    if Path(f'{datapath}/{idx}.npy').is_file(): 
        if shared: pba.update.remote(1)
        return

    seed = idx + 100000 if test else idx
    rng = np.random.default_rng(seed)
    
    if lab_setup:
        empty_pos = [4,7]
        if orig:
            filled_pos = [
                [0, 0], [3, 0], [4, 0], [6, 0], [8, 0], [9, 0], [10, 0],
                [11, 0], [0, 1], [1, 1], [3, 1], [4, 1], [5, 1], [6, 1],
                [7, 1], [9, 1], [10, 1], [11, 1], [1, 2], [2, 2], [3, 2],
                [4, 2], [5, 2], [6, 2], [8, 2], [9, 2], [11, 2], [0, 3],
                [1, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3],
                [9, 3], [10, 3], [0, 4], [1, 4], [2, 4], [3, 4], [8, 4],
                [9, 4], [10, 4], [11, 4], [0, 5], [1, 5], [8, 5], [9, 5],
                [10, 5], [11, 5], [1, 6], [2, 6], [8, 6], [10, 6], [11, 6],
                [0, 7], [2, 7], [3, 7], [8, 7], [9, 7], [10, 7], [0, 8],
                [1, 8], [2, 8], [3, 8], [4, 8], [7, 8], [8, 8], [9, 8],
                [11, 8], [0, 9], [1, 9], [2, 9], [3, 9], [5, 9], [6, 9],
                [7, 9], [8, 9], [9, 9], [10, 9], [11, 9], [1, 10], [4, 10],
                [5, 10], [7, 10], [8, 10], [10, 10], [0, 11], [1, 11], [3, 11],
                [5, 11], [6, 11], [8, 11], [9, 11], [10, 11], [11, 11],
            ]
            filled_pos = [[pos[0], pos[1], 0] for pos in filled_pos]
            a = 0
            b = 0.5
            c = 1
            d = 1.5
            mag_angles = [
                d,c,b,d,b,a,c,b,
                b,c,d,c,b,a,b,b,d,c,
                a,d,a,c,c,b,d,b,a,
                d,a,b,c,b,b,d,b,c,c,
                d,b,a,b,b,d,c,b,
                c,c,a,d,c,d,
                d,d,c,c,c,            
                c,a,c,c,a,b,
                a,a,c,d,c,d,c,a,c,
                b,d,d,a,c,d,d,c,b,d,a,
                c,d,d,c,d,c,
                d,c,d,c,d,a,d,b,c,
            ]
            rand_arr = rng.random(size=(len(filled_pos),1))
            mag_angles = [[np.pi/2 - np.pi/36 + np.pi/18 * rand_arr[i,0], np.pi * azi]
                          for i, azi in enumerate(mag_angles)]

        else:
            A = rng.choice([0, 1], size=(12,12,1), p=[0.25, 0.75])
            filled_pos = [[i, j, 0] for i in range(12) for j in range(12) if A[i][j] == 1 and
                (i < empty_pos[0] or i > empty_pos[1] or j < empty_pos[0] or j > empty_pos[1])]    
            rand_arr = rng.random(size=(len(filled_pos),2))
            mag_angles = [[np.pi/2 - np.pi/36 + np.pi/18 * rand_arr[i,0],
                          2 * np.pi * rand_arr[i,1]] for i in range(len(filled_pos))]
    
        x_places = 12
        x_area = 8.4
        z_places = 1
        z_area = 0.7
        gap_l = 0.4
        gap_r = 0.3

    else:
        x_places = 10
        x_area = 1
        z_places = 5
        z_area = 0.5
        gap_l = 0.05
        gap_r = 0.05
        hole_dict = {0:[4,5], 1:[4,6], 2:[3,6], 3:[3,7]}
        empty_pos = hole_dict[rng.integers(4)]
        A = rng.integers(2, size=(10,10,z_places))
        filled_pos = [[i, j, k] for i in range(10) for j in range(10) for k in range(z_places) 
            if A[i][j][k] == 1 and (i < empty_pos[0] or i > empty_pos[1] or j < empty_pos[0]
            or j > empty_pos[1] or k < 2 or k > 2)]
        rand_arr = rng.random(size=(len(filled_pos),2))
        mag_angles = [[np.pi * rand_arr[i,0], 2 * np.pi * rand_arr[i,1]]
                      for i in range(len(filled_pos))]
    
    (tiles, _, _) = magtense.setup(
        places=[x_places, x_places, z_places],
        area=[x_area, x_area, z_area],
        mag_angles=mag_angles,
        filled_positions=filled_pos
    )

    empty_pos[0] *= (x_area / x_places)
    empty_pos[1] *= (x_area / x_places)
    
    # Area to evaluate field in
    x_eval = np.linspace(empty_pos[0] + gap_l, empty_pos[1] + gap_r, res + 1)
    y_eval = np.linspace(empty_pos[0] + gap_l, empty_pos[1] + gap_r, res + 1)
    if ext:
        res_z = 3
        z_eval = np.linspace(-(empty_pos[1] - empty_pos[0]) / res, 
            (empty_pos[1] - empty_pos[0]) / res, res_z) + z_area / 2
        xv, yv, zv = np.meshgrid(x_eval[:res], y_eval[:res], z_eval)
    else:
        xv, yv = np.meshgrid(x_eval[:res], y_eval[:res])
        zv = np.zeros(res * res) + z_area / 2
    pts_eval = np.hstack([xv.reshape(-1,1), yv.reshape(-1,1), zv.reshape(-1,1)])

    # Running simulation
    iterated_tiles = magtense.iterate_magnetization(tiles)
    N = magtense.get_N_tensor(iterated_tiles, pts_eval)
    H = magtense.get_H_field(iterated_tiles, pts_eval, N)

    if ext:
        # Tensor image with shape CxHxWxD
        field = np.zeros(shape=(3, res, res, res_z), dtype=np.float32)
        field = H.reshape((res,res,res_z,3)).transpose((3,0,1,2))
    else:
        # Tensor image with shape CxHxW
        field = np.zeros(shape=(3, res, res), dtype=np.float32)
        field = H.reshape((res,res,3)).transpose((2,0,1))

    # Saving field in [T]
    field = field * 4 * np.pi * 1e-7

    # Plot first ten samples
    if check and idx < 20:
        v_max = 0.025 if lab_setup else 0.2
        filename = f'{t_start.strftime("%y%m%d_%H%M")}_{idx}'
        sample_check(field, v_max=v_max, filename=filename, cube=ext)
    
    if field is not None:
        np.save(f'{datapath}/{idx}.npy', field)
    
    # Progress bar
    if shared: pba.update.remote(1)


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


def sample_check(field, v_max=1, filename=f'foo_{datetime.utcnow().strftime("%y%m%d_%H%M")}', cube=False):
    plotpath = Path(__file__).parent.resolve() / '..' / 'plots' / 'sample_check'
    if not plotpath.exists(): plotpath.mkdir(parents=True)
    plt.clf()
    labels = ['Bx-field', 'By-field', 'Bz-field']
    nrows = 3 if cube else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=3, sharex=True, sharey=True, figsize=(15,10))

    if cube:
        for i, z in enumerate([0, 1, 2]):
            for j, comp in enumerate(field[:,:,:,z]):
                ax = axes.flat[i * 3 + j]
                im = ax.imshow(comp, cmap='bwr',
                    norm=colors.Normalize(vmin=-v_max, vmax=v_max), origin="lower")
                ax.set_title(labels[j] + f'@{z+1}')

    else:
        for i, comp in enumerate(field):
            ax = axes.flat[i]
            im = ax.imshow(comp, cmap='bwr',
                norm=colors.Normalize(vmin=-v_max, vmax=v_max), origin="lower")
            ax.set_title(labels[i])
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.345, 0.015, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f'{plotpath}/{filename}.png', bbox_inches='tight')


def create_db(size, name='', res=256, num_proc=None, check=False, ext=False,
              test=False, lab_setup=False, orig=False, start_idx=0):
    datapath = Path(__file__).parent.resolve() / '..' / 'data' / f'{name}_{res}'
    if not datapath.exists(): datapath.mkdir(parents=True)
    worker = cpu_count() if num_proc is None else num_proc
    if worker > 1:
        ray.init(num_cpus=worker, include_dashboard=False, local_mode=False)

    t_start = datetime.utcnow()
    print(f'[INFO {t_start.strftime("%d/%m %H:%M:%S")}] #Data: {size}'
          +  f' | #Worker: {worker} | #Path: {datapath}')

    if num_proc == 1:
        for idx in range(start_idx, size):
            generate_exp(idx, res, datapath, t_start, check, ext, test, lab_setup, orig)
    else:
        pb = ProgressBar(size - start_idx)
        actor = pb.actor
        gen_exp_ray = ray.remote(generate_exp)
        res = [gen_exp_ray.remote(idx, res, datapath, t_start, check, ext, test, lab_setup, orig, True, actor)
               for idx in range(start_idx, size)]
        pb.print_until_done()
        _ = [ray.get(r) for r in res]
        ray.shutdown()


if __name__ == '__main__':
    create_db(
        size=20,
        name='viz_test_ext',
        res=256,
        num_proc=4,
        # check=True,
        ext=True,
        # lab_setup=True,
        # orig=True,
        test=True
    )
    