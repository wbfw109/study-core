# Wrriten at 📅 2024-09-18 19:58:21
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


class ModelManager:
    def __init__(
        self, model_dir: Optional[Path] = None, executable_dir: Optional[Path] = None
    ):
        """
        Initialize the model and executable manager.

        Args:
            model_dir (Optional[Path], optional): Directory path where models will be stored. Defaults to ~/ai_models.
            executable_dir (Optional[Path], optional): Directory where executables will be stored. Defaults to ~/ai_models/executables.
        """
        # Set model directory (default: ~/ai_models)
        if model_dir is None:
            model_dir = Path.home() / "ai_models"
        self.model_dir: Path = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Set executable directory (default: ~/ai_models/executables)
        if executable_dir is None:
            executable_dir = Path.home() / "ai_models" / "executables"
        self.executable_dir: Path = executable_dir
        self.executable_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store models and executables
        self.models: dict[str, dict[str, dict[str, str]]] = {}
        self.executables: dict[str, dict[str, dict[str, str]]] = {}

    # ====================== MODEL MANAGEMENT =========================== #
    def register_model(self, source: str, model_name: str, model_url: str) -> None:
        """
        Register a new model from a specific source (e.g., RealESRGAN).

        Args:
            source (str): The source of the model (e.g., 'RealESRGAN').
            model_name (str): The name of the model.
            model_url (str): The URL to download the model.
        """
        model_filename = model_url.split("/")[-1]
        model_path = self.model_dir / model_filename

        if source not in self.models:
            self.models[source] = {}

        # Register model with its URL and calculated path
        self.models[source][model_name] = {
            "url": model_url,
            "path": str(model_path),  # Save path as string
        }

    def download_and_update_model_paths(
        self, source: str, overwrite: Optional[str] = "ignore"
    ) -> None:
        """
        Download all models from a registered source and update their paths.
        If the model already exists, only the path will be updated.

        Args:
            source (str): The source of the models to download (e.g., 'RealESRGAN').
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        if source not in self.models:
            raise ValueError(f"No models registered under source '{source}'.")

        for model_name in self.models[source]:
            self.download_model(source, model_name, overwrite)

    def download_model(
        self, source: str, model_name: str, overwrite: Optional[str] = "ignore"
    ) -> Path:
        """
        Download a specific model from a registered source and grant execute permissions.

        Args:
            source (str): The source of the model (e.g., 'RealESRGAN').
            model_name (str): The name of the model.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.

        Returns:
            Path: The path where the downloaded model is saved.
        """
        if source not in self.models or model_name not in self.models[source]:
            raise ValueError(f"Invalid source '{source}' or model name '{model_name}'.")

        model_info = self.models[source][model_name]
        model_url: str = model_info["url"]
        model_path: Path = Path(model_info["path"])

        if model_path.exists() and overwrite == "ignore":
            print(
                f"Model '{model_name}' already exists at {model_path}. Skipping download."
            )
            return model_path
        elif model_path.exists() and overwrite == "delete":
            model_path.unlink()

        command = ["wget", model_url, "-O", str(model_path)]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Model '{model_name}' downloaded and saved to {model_path}")
        else:
            raise RuntimeError(f"Error downloading the model: {result.stderr}")

        # Grant execute permissions to the model file
        model_path.chmod(
            0o755
        )  # Read, write, and execute for the owner; read and execute for others

        return model_path

    def list_model_paths(self, source: Optional[str] = None) -> dict[str, str]:
        """
        List the paths of all registered models, either from a specific source or from all sources.

        Args:
            source (Optional[str], optional): The source of the models to list paths for. If None, lists all models.

        Returns:
            dict: A dictionary of model names and their paths for the specified source or all sources.
        """
        if source:
            if source not in self.models:
                raise ValueError(f"Invalid source '{source}'.")
            return {
                model: self.models[source][model]["path"]
                for model in self.models[source]
            }
        return {
            model: self.models[source_name][model]["path"]
            for source_name in self.models
            for model in self.models[source_name]
        }

    # ====================== EXECUTABLE MANAGEMENT =========================== #

    def register_executable(
        self, source: str, executable_name: str, executable_url: str
    ) -> None:
        """
        Register a new executable from a specific source.

        Args:
            source (str): The source of the executable (e.g., 'RealESRGAN').
            executable_name (str): The name of the executable.
            executable_url (str): The URL to download the executable.
        """
        executable_filename = executable_url.split("/")[-1]
        executable_path = self.executable_dir / executable_filename

        if source not in self.executables:
            self.executables[source] = {}

        self.executables[source][executable_name] = {
            "url": executable_url,
            "path": str(executable_path),  # Initially pointing to the archive
        }

    def download_and_update_executable_paths(
        self, source: str, overwrite: Optional[str] = "ignore"
    ) -> None:
        """
        Download all executables from a registered source and update their paths.
        If the executable already exists, only the path will be updated.

        Args:
            source (str): The source of the executables to download (e.g., 'RealESRGAN').
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        if source not in self.executables:
            raise ValueError(f"No executables registered under source '{source}'.")

        for executable_name in self.executables[source]:
            self.download_and_update_executable_path(source, executable_name, overwrite)

    def download_and_update_executable_path(
        self, source: str, executable_name: str, overwrite: Optional[str] = "ignore"
    ) -> Path:
        """
        Download and extract a specific executable from a registered source, and update the path
        to point to the actual executable (not the archive). Also grants execute permissions.

        Args:
            source (str): The source of the executable.
            executable_name (str): The name of the executable.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.

        Returns:
            Path: The path to the main executable file.
        """
        if (
            source not in self.executables
            or executable_name not in self.executables[source]
        ):
            raise ValueError(
                f"Invalid source '{source}' or executable name '{executable_name}'."
            )

        executable_info = self.executables[source][executable_name]
        executable_url: str = executable_info["url"]
        executable_archive_path: Path = Path(executable_info["path"])

        extract_dir = self.executable_dir / executable_name

        if extract_dir.exists() and overwrite == "ignore":
            print(
                f"Executable '{executable_name}' already exists at {extract_dir}. Skipping download."
            )
            main_executable = self.find_main_executable(extract_dir)
            self.executables[source][executable_name]["path"] = str(main_executable)
            return main_executable

        elif extract_dir.exists() and overwrite == "delete":
            shutil.rmtree(extract_dir, ignore_errors=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / executable_url.split("/")[-1]
            command = ["wget", executable_url, "-O", str(tmp_path)]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Error downloading the executable: {result.stderr}")

            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        main_executable = self.find_main_executable(extract_dir)
        self.executables[source][executable_name]["path"] = str(main_executable)

        # Grant execute permissions to the main executable
        main_executable.chmod(
            0o755
        )  # Read, write, and execute for the owner; read and execute for others

        return main_executable

    def find_main_executable(self, directory: Path) -> Path:
        """
        Find the main executable file in a given directory (file without an extension).

        Args:
            directory (Path): The directory to search for the main executable.

        Returns:
            Path: The path to the main executable file.
        """
        for item in directory.iterdir():
            if item.is_file() and not item.suffix:
                return item

        raise FileNotFoundError(f"No main executable found in directory {directory}")

        raise FileNotFoundError(f"No main executable found in directory {directory}")

    def list_executable_paths(self, source: Optional[str] = None) -> dict[str, str]:
        """
        List the paths of all registered executables, either from a specific source or from all sources.

        Args:
            source (Optional[str], optional): The source of the executables to list paths for. If None, lists all executables.

        Returns:
            dict: A dictionary of executable names and their paths for the specified source or all sources.
        """
        if source:
            if source not in self.executables:
                raise ValueError(f"Invalid source '{source}'.")
            return {
                exe: self.executables[source][exe]["path"]
                for exe in self.executables[source]
            }
        return {
            exe: self.executables[source_name][exe]["path"]
            for source_name in self.executables
            for exe in self.executables[source_name]
        }

    # ====================== DOWNLOAD ALL (FOR BOTH MODELS AND EXECUTABLES) =========================== #

    def download_all(
        self, source: Optional[str] = None, overwrite: Optional[str] = "ignore"
    ) -> None:
        """
        Download all models and executables from a registered source or from all sources.

        Args:
            source (Optional[str], optional): The source of the models and executables to download (e.g., 'RealESRGAN').
                                              If None, downloads all models and executables from all sources.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        # Download all models for a specific source or all sources
        if source:
            if source in self.models:
                for model_name in self.models[source]:
                    self.download_model(source, model_name, overwrite)
            if source in self.executables:
                for executable_name in self.executables[source]:
                    self.download_and_update_executable_path(
                        source, executable_name, overwrite
                    )
        else:
            for source_name in self.models:
                for model_name in self.models[source_name]:
                    self.download_model(source_name, model_name, overwrite)
            for source_name in self.executables:
                for executable_name in self.executables[source_name]:
                    self.download_and_update_executable_path(
                        source_name, executable_name, overwrite
                    )


class RealESRGANPlugin:
    """
    RealESRGAN Plugin to register and manage models and executables for RealESRGAN using the ModelManager.

    This plugin allows users to manage only RealESRGAN models and executables,
    including downloading and updating their paths.

    Example Usage:
    --------------
    >>> model_manager = ModelManager()
    >>> real_esrgan_plugin = RealESRGANPlugin(model_manager)

    # List RealESRGAN models
    >>> print(real_esrgan_plugin.list_real_esrgan_models())

    # Download and update all RealESRGAN models
    >>> real_esrgan_plugin.download_and_update_real_esrgan_model_paths(overwrite="ignore")

    # Get model path for a specific model
    >>> model_path = real_esrgan_plugin.get_model_path("realesr-general-x4v3")
    >>> print(f"Model path: {model_path}")

    # List all RealESRGAN executables
    >>> print(real_esrgan_plugin.list_real_esrgan_executables())

    # Download and update all RealESRGAN executables
    >>> real_esrgan_plugin.download_and_update_real_esrgan_executable_paths(overwrite="ignore")

    # Get executable path for a specific executable
    >>> executable_path = real_esrgan_plugin.get_executable_path("realesrgan-ncnn-vulkan")
    >>> print(f"Executable path: {executable_path}")
    """

    def __init__(self, manager: ModelManager):
        """
        Initialize the RealESRGAN plugin with the provided ModelManager instance.

        Args:
            manager (ModelManager): An instance of the ModelManager to manage models and executables.
        """
        self.manager = manager

        # Register Real-ESRGAN models
        self.manager.register_model(
            "RealESRGAN",
            "realesr-general-x4v3",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        )
        self.manager.register_model(
            "RealESRGAN",
            "realesr-general-wdn-x4v3",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        )

        # Register Real-ESRGAN executables
        self.manager.register_executable(
            "RealESRGAN",
            "realesrgan-ncnn-vulkan",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        )

    # ===================== MODEL MANAGEMENT ====================== #

    def download_and_update_real_esrgan_model_paths(
        self, overwrite: Optional[str] = "ignore"
    ) -> None:
        """
        Download all registered RealESRGAN models and update their paths.
        If the model already exists, only the path will be updated.

        Args:
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        self.manager.download_and_update_model_paths("RealESRGAN", overwrite)

    def list_real_esrgan_models(self) -> dict[str, str]:
        """
        List all registered RealESRGAN models with their paths.

        Returns:
            dict[str, str]: A dictionary of RealESRGAN model names and their file paths.
        """
        return self.manager.list_model_paths("RealESRGAN")

    def get_model_path(self, model_name: str) -> str:
        """
        Retrieve the full path of a registered RealESRGAN model from the ModelManager.

        Args:
            model_name (str): The name of the model (e.g., "realesr-general-x4v3").

        Returns:
            str: The file path of the requested model.
        """
        model_paths = self.manager.list_model_paths("RealESRGAN")
        if model_name in model_paths:
            return model_paths[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found.")

    # ===================== EXECUTABLE MANAGEMENT ====================== #

    def download_and_update_real_esrgan_executable_paths(
        self, overwrite: Optional[str] = "ignore"
    ) -> None:
        """
        Download all registered RealESRGAN executables and update their paths.
        If the executable already exists, only the path will be updated.

        Args:
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        self.manager.download_and_update_executable_paths("RealESRGAN", overwrite)

    def list_real_esrgan_executables(self) -> dict[str, str]:
        """
        List all registered RealESRGAN executables with their paths.

        Returns:
            dict[str, str]: A dictionary of RealESRGAN executable names and their file paths.
        """
        return self.manager.list_executable_paths("RealESRGAN")

    def get_executable_path(self, executable_name: str) -> str:
        """
        Retrieve the full path of a registered RealESRGAN executable from the ModelManager.

        Args:
            executable_name (str): The name of the executable (e.g., "realesrgan-ncnn-vulkan").

        Returns:
            str: The file path of the requested executable.
        """
        executable_paths = self.manager.list_executable_paths("RealESRGAN")
        if executable_name in executable_paths:
            return executable_paths[executable_name]
        else:
            raise ValueError(f"Executable '{executable_name}' not found.")
