from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    openfoam_dir: str
    n_procs: int
