# Wrriten at 📅 2024-09-18 12:53:06
import subprocess
from pathlib import Path
from typing import Optional


class ModelManager:
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the model manager.

        Args:
            model_dir (Optional[Path], optional): Directory path where models will be stored. Defaults to ~/ai_models.
        """
        # Set model directory (default: ~/ai_models)
        if model_dir is None:
            model_dir = Path.home() / "ai_models"
        self.model_dir: Path = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store models from various sources
        self.models: dict[str, dict[str, dict[str, str]]] = {}

    def register_model(self, source: str, model_name: str, model_url: str) -> None:
        """
        Register a new model from a specific source (e.g., RealESRGAN).

        Args:
            source (str): The source of the model (e.g., 'RealESRGAN').
            model_name (str): The name of the model.
            model_url (str): The URL to download the model.
        """
        # Extract the filename from the model URL (the part after the last '/')
        model_filename = model_url.split("/")[-1]
        model_path = self.model_dir / model_filename

        if source not in self.models:
            self.models[source] = {}

        # Register model with its URL and calculated path
        self.models[source][model_name] = {
            "url": model_url,
            "path": str(model_path),  # Save path as string
        }

    def download_model(
        self, source: str, model_name: str, overwrite: Optional[str] = "ignore"
    ) -> Path:
        """
        Download a specific model from a registered source.

        Args:
            source (str): The source of the model (e.g., 'RealESRGAN').
            model_name (str): The name of the model.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.

        Raises:
            ValueError: If the source or model name is invalid.

        Returns:
            Path: The path where the downloaded model is saved.
        """
        if source not in self.models or model_name not in self.models[source]:
            raise ValueError(f"Invalid source '{source}' or model name '{model_name}'.")

        model_info = self.models[source][model_name]
        model_url: str = model_info["url"]
        model_path: Path = Path(model_info["path"])

        # Handle existing file based on the 'overwrite' option
        if model_path.exists():
            if overwrite == "ignore":
                print(
                    f"Model '{model_name}' already exists at {model_path}. Skipping download."
                )
                return model_path
            elif overwrite == "overwrite":
                print(f"Overwriting existing model '{model_name}' at {model_path}.")
            elif overwrite == "delete":
                print(
                    f"Deleting existing model '{model_name}' at {model_path} and downloading again."
                )
                model_path.unlink()  # Delete the existing file
            else:
                raise ValueError(
                    f"Invalid overwrite option: {overwrite}. Use 'ignore', 'overwrite', or 'delete'."
                )

        # Use wget to download the model
        command = ["wget", model_url, "-O", str(model_path)]
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if download was successful
        if result.returncode == 0:
            print(f"Model '{model_name}' downloaded and saved to {model_path}")
        else:
            raise RuntimeError(f"Error downloading the model: {result.stderr}")

        return model_path

    def download_all(
        self, source: Optional[str] = None, overwrite: Optional[str] = "ignore"
    ) -> None:
        """
        Download all models from a registered source or all sources.

        Args:
            source (Optional[str], optional): The source of the models to download (e.g., 'RealESRGAN').
                                              If None, downloads all models from all sources.
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        if source:
            if source not in self.models:
                raise ValueError(f"Invalid source '{source}'.")
            for model_name in self.models[source]:
                self.download_model(source, model_name, overwrite)
        else:
            for source_name in self.models:
                for model_name in self.models[source_name]:
                    self.download_model(source_name, model_name, overwrite)

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


class RealESRGANPlugin:
    """
    RealESRGAN Plugin to register models with the general ModelManager.

    Example Usage:
    --------------
    >>> model_manager = ModelManager()
    >>> real_esrgan_plugin = RealESRGANPlugin(model_manager)
    >>> print(real_esrgan_plugin.list_real_esrgan_models())
    >>> real_esrgan_plugin.download_real_esrgan_models(overwrite="ignore")
    >>> print(model_manager.list_model_paths())
    """

    def __init__(self, manager: ModelManager):
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

    def list_real_esrgan_models(self) -> dict[str, str]:
        """
        List all registered RealESRGAN models with their paths.

        Returns:
            dict[str, str]: A dictionary of RealESRGAN model names and their file paths.
        """
        return self.manager.list_model_paths("RealESRGAN")

    def download_real_esrgan_models(self, overwrite: Optional[str] = "ignore") -> None:
        """
        Download all registered RealESRGAN models.

        Args:
            overwrite (Optional[str], optional): Determines what to do if the file already exists.
                                                 Options are 'ignore' (default), 'overwrite', or 'delete'.
        """
        self.manager.download_all("RealESRGAN", overwrite)

    def get_manager(self) -> ModelManager:
        """
        Returns the ModelManager instance, so that other plugins can be used.

        Returns:
            ModelManager: The instance of the model manager.
        """
        return self.manager
