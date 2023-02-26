from django.apps import AppConfig


class ml_modelConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ml_model"

    def ready(self):
        from ml_model.global_object import GlobalDeivceCamera, GlobalConfig

        GlobalDeivceCamera.initialize_dict()

        # in schedular
        # GlobalDeivceCamera.set_connect_status_dict_from_connect_threshold_seconds(GlobalConfig.DEVICE_CUBE_NETWORK_STATUS_CYCLE_SECONDS)

        pass
