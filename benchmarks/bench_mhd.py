import jax
import jax.numpy as jnp
import adirondax as adx
from adirondax.hydro.common2d import get_curl, get_avg

# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class MHDSuite:
    """
    A benchmark for MHD.
    """

    def setup(self):
        n = 512
        nt = 10 * int(n / 32)
        t_stop = 0.05
        dt = t_stop / nt
        gamma = 5.0 / 3.0
        box_size = 1.0

        self.params = {
            "physics": {
                "hydro": True,
                "magnetic": True,
                "quantum": False,
                "gravity": False,
            },
            "mesh": {
                "type": "cartesian",
                "resolution": [n, n],
                "boxsize": [box_size, box_size],
            },
            "simulation": {
                "stop_time": t_stop,
                "timestep": dt,
                "n_timestep": nt,
            },
            "hydro": {
                "eos": {"type": "ideal", "gamma": gamma},
            },
        }

    def run_sim(self):
        gamma = self.params["hydro"]["eos"]["gamma"]
        box_size = self.params["mesh"]["boxsize"][0]
        n = self.params["mesh"]["resolution"][0]
        dx = box_size / n
        sim = adx.Simulation(self.params)
        sim.state["t"] = jnp.array(0.0)
        X, Y = sim.mesh
        sim.state["rho"] = (gamma**2 / (4.0 * jnp.pi)) * jnp.ones(X.shape)
        sim.state["vx"] = -jnp.sin(2.0 * jnp.pi * Y)
        sim.state["vy"] = jnp.sin(2.0 * jnp.pi * X)
        P_gas = (gamma / (4.0 * jnp.pi)) * jnp.ones(X.shape)
        # (Az is at top-right node of each cell)
        xlin_node = jnp.linspace(dx, box_size, n)
        Xn, Yn = jnp.meshgrid(xlin_node, xlin_node, indexing="ij")
        Az = jnp.cos(4.0 * jnp.pi * Xn) / (
            4.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi)
        ) + jnp.cos(2.0 * jnp.pi * Yn) / (2.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi))
        bx, by = get_curl(Az, dx)
        Bx, By = get_avg(bx, by)
        P_tot = P_gas + 0.5 * (Bx**2 + By**2)
        sim.state["P"] = P_tot
        sim.state["bx"] = bx
        sim.state["by"] = by
        sim.run()
        return jnp.mean(sim.state["bx"] ** 2)

    def time_run_sim(self):
        self.run_sim()

    def peakmem_run_sim(self):
        self.run_sim()
