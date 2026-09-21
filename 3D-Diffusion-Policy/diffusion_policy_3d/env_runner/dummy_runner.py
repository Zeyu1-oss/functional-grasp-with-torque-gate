from diffusion_policy_3d.env_runner.base_runner import BaseRunner


class DummyRunner(BaseRunner):
    def run(self, policy, **kwargs):
        return {}
