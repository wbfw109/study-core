import datetime
from typing import Optional, Union
from django.db import models
from django.db.models.query import QuerySet
from django.db.models.query_utils import Q
from django.utils import timezone
from ml_model.model.base import ThreatEvent, ConnectStatus
from ml_model.model.device_cube import DeviceCube
import time


class DeviceCamera(models.Model):
    """
    + about ip_address
        - it includes Port and Network mask
        - e.g. "554" is port and "11" is Network mask in "10.10.10.26:554/11"
    """
    id: int = models.AutoField(primary_key=True)
    cube: DeviceCube = models.ForeignKey(
        DeviceCube, on_delete=models.DO_NOTHING, null=True
    )
    ip_address: str = models.CharField(max_length=100)
    device_id: str = models.CharField(max_length=50)
    device_pw: str = models.CharField(max_length=50)
    location: str = models.TextField()
    is_deleted: bool = models.BooleanField(blank=True, default=False)
    created_datetime: datetime.datetime = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "device_camera"

    def __init__(self, *args, **kwargs):
        super(DeviceCamera, self).__init__(*args, **kwargs)
        self._connect_status: Optional[ConnectStatus] = None

    @property
    def rtsp_url(self):
        return fr"rtsp://{self.device_id}:{self.device_pw}@{self.ip_address}"

    @property
    def connect_status(self):
        """
        + Note
            It must be managed by ml_model.global_object.GlobalDeviceCamera
        """
        return self._connect_status

    @connect_status.setter
    def connect_status(self, connect_status: ConnectStatus):
        """
        + about ConnectStatus.modifed_time
            - if ConnectStatus.modifed_time is None, it considered as not connected.
            - so ConnectStatus.modifed_time must not be updated.

        Args:
            connect_status (ConnectStatus): [description]
        """
        if ConnectStatus.modified_time is None:
            self._connect_status.is_connected = connect_status.is_connected
        else:
            self._connect_status = connect_status


class DeviceCameraEvent(models.Model):
    id: int = models.AutoField(primary_key=True)
    camera: DeviceCamera = models.ForeignKey(DeviceCamera, on_delete=models.DO_NOTHING)
    threat_event: ThreatEvent = models.ForeignKey(
        ThreatEvent, on_delete=models.DO_NOTHING
    )
    image: str = models.ImageField(upload_to="device_camera_event", null=True)
    created_timestamp: int = models.PositiveBigIntegerField(default=int(time.time()))

    class Meta:
        db_table = "device_camera_event"


class DeviceCameraConnectStatus(models.Model):
    id: int = models.AutoField(primary_key=True)
    device_cube: DeviceCamera = models.ForeignKey(DeviceCube, on_delete=models.DO_NOTHING)
    device_camera: DeviceCamera = models.ForeignKey(DeviceCamera, on_delete=models.CASCADE)
    modified_time: int = models.PositiveBigIntegerField(default=int(time.time()))
    is_connected: bool = models.BooleanField(default=False)

    class Meta:
        db_table = "device_camera_connect_status"

    @staticmethod
    def set_last_connect_status_dict_to_database(
        connect_status_dict: dict[int, dict[int, ConnectStatus]]
    ) -> None:
        DeviceCameraConnectStatus.objects.all().delete()
        deivce_camera_connect_status_list: list[DeviceCameraConnectStatus] = []
        for device_cube_id, device_camera_dict in connect_status_dict.items():
            for device_camera_id, connect_status in device_camera_dict.items():
                deivce_camera_connect_status_list.append(
                    DeviceCameraConnectStatus(
                        cube_id=device_cube_id,
                        camera_id=device_camera_id,
                        modified_time=connect_status.modified_time,
                        is_connected=connect_status.is_connected,
                    )
                )
        DeviceCameraConnectStatus.save()

    @staticmethod
    def get_last_connect_status_dict_from_database(
        in_filter_cube_id_list: list[int] = None,
        in_filter_camera_id_list: list[int] = None,
    ) -> dict[int, dict[int, ConnectStatus]]:
        q_list: list[Q] = []
        if in_filter_cube_id_list:
            q_list.append(Q(cube_id__in=in_filter_cube_id_list))
        if in_filter_camera_id_list:
            q_list.append(Q(camera_id__in=in_filter_camera_id_list))

        device_camera_connect_status_list: list[
            Union[QuerySet, DeviceCameraConnectStatus]
        ] = DeviceCameraConnectStatus.objects.filter(*q_list).all()
        device_camera_connect_status_dict: dict[int, dict[int, ConnectStatus]] = {}
        for device_camera_connect_status in device_camera_connect_status_list:
            device_camera_connect_status_dict[device_camera_connect_status.cube_id] = {}
        for device_camera_connect_status in device_camera_connect_status_list:
            device_camera_connect_status_dict[device_camera_connect_status.cube_id][
                device_camera_connect_status.camera_id
            ] = ConnectStatus(
                modified_time=device_camera_connect_status.modified_time,
                is_connected=device_camera_connect_status.is_connected,
            )
        return device_camera_connect_status_dict
