from __future__ import annotations
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile
from django.db import transaction
from django.db.models.aggregates import Max
from django.db.models.query import QuerySet
from io import BytesIO
from ml_model.model.base import ConnectStatus, ThreatEvent
from ml_model.model.device_camera import (
    DeviceCamera,
    DeviceCameraEvent,
    DeviceCameraConnectStatus,
)
from ml_model.model.device_cube import DeviceCube
from PIL import Image
from types import SimpleNamespace
from typing import TypedDict, Union
import base64
import dataclasses
import datetime
import imghdr
import logging
import pytz
import time
import json


class GlobalConfig:
    DEVICE_CUBE_NETWORK_STATUS_CYCLE_SECONDS: int = 3600
    DEVICE_CAMERA_EVENT_NAME_PREFIX: str = "device_camera_event-"
    # DEVICE_CAMERA_EVENT_NONE_IMAGE: str = development_settings.BASE_DIR
    DEVICE_CAMERA_EVENT_NONE_IMAGE: str = None
    DEVICE_CAMERA_EVENT_INDEX_THAT_IS_NORMAL: int = 1
    DEVICE_CAMERA_EVENT_INDEX_THAT_IS_CAMERA_ERROR: int = 4


class GlobalResponseMessage:
    SUCCESS: int = 0
    ARGUMENTS_ERROR_TO_VALIDATE_SERIAL_NUMBER :str  = r"Require for serial number."
    ARGUMENTS_ERROR_TO_SET_DEVICE_CUBE_NAME :str  = r"Require for name."
    ARGUMENTS_ERROR_TO_REGISTER_DEVICE_CUBE :str  = r"Require keys {id: int, firmware: str}."
    UNKNOWN_SERIAL_NUMBER :str  = r"Unknown or already registered serial numbers exist."
    ARGUMENTS_ERROR_SET_DEVICE_CUBE_FIRMWARE :str  = r"Require keys {id: int, firmware: str}."
    ARGUMENTS_ERROR_TO_SET_DEVICE_CUBE_NAME :str  = r"Require keys {id: int, name: str}."
    ARGUMENTS_ERROR_TO_SET_DEVICE_CUBE: str = r"Require keys {id: int} and one of {ip_address: str, device_id: str, device_pw: str, location: str, cube_id: int}"

class GlobalDeivceCamera:
    # connect_status_dict[cube_id][camera_id] = ConnectStatus()
    connect_status_dict: dict[int, dict[int, ConnectStatus]] = {}
    # most_recent_device_camera_event_image_dict[camera_id] = image
    most_recent_device_camera_event_image_dict: dict[int, str] = {}

    _is_initialized_for_last_connected_time_dict: bool = False

    @classmethod
    def initialize_dict(cls) -> None:
        """It is idempotent method."""
        if not cls._is_initialized_for_last_connected_time_dict:
            valid_device_camera_list: list[Union[QuerySet, dict[str, int]]] = (
                DeviceCamera.objects.select_related("cube")
                .filter(is_deleted=False, cube__is_deleted=False)
                .values("id", "cube_id")
            )

            if DeviceCameraConnectStatus.objects.all().count() > 0:
                cls.connect_status_dict = DeviceCameraConnectStatus.get_last_connect_status_dict_from_database(
                    in_filter_camera_id_list=[
                        x["cube_id"] for x in valid_device_camera_list
                    ],
                    in_filter_cube_id_list=[x["id"] for x in valid_device_camera_list],
                )
            else:
                for valid_device_camera in valid_device_camera_list:
                    cls.connect_status_dict[valid_device_camera["cube_id"]] = {}
                for valid_device_camera in valid_device_camera_list:
                    cls.connect_status_dict[valid_device_camera["cube_id"]][
                        valid_device_camera["id"]
                    ] = ConnectStatus()

            most_recent_device_camera_event_dict: dict[
                int, Union[QuerySet, dict[str]]
            ] = {}
            for device_camera_event in (
                DeviceCameraEvent.objects.filter(
                    camera_id__in=[x["id"] for x in valid_device_camera_list]
                )
                .values("camera_id", "image")
                .annotate(most_recent_created_timestamp=Max("created_timestamp"))
            ):
                most_recent_device_camera_event_dict[
                    device_camera_event["camera_id"]
                ] = device_camera_event["image"]
            for valid_device_camera in valid_device_camera_list:
                cls.most_recent_device_camera_event_image_dict[
                    valid_device_camera["id"]
                ] = GlobalConfig.DEVICE_CAMERA_EVENT_NONE_IMAGE
            for device_camera_id, image in most_recent_device_camera_event_dict.items():
                if image:
                    cls.most_recent_device_camera_event_image_dict[
                        device_camera_id
                    ] = image
                else:
                    cls.most_recent_device_camera_event_image_dict[
                        device_camera_id
                    ] = GlobalConfig.DEVICE_CAMERA_EVENT_NONE_IMAGE

            cls._is_initialized_for_last_connected_time_dict = True

    @classmethod
    def set_connect_status_dict_from_device_cube_id(
        cls,
        device_cube_id: int,
        connect_status: ConnectStatus = ConnectStatus(
            modified_time=None, is_connected=None
        ),
    ) -> None:
        """
        It is useful when Django web server can not connect ml_model Cube program.

        + Warning
            - following items must be verified before using this function
                - ObjectDoesNotExist Exception from device_cube
                - device_cube.is_deleted False
        """
        valid_device_camera_list: list[
            Union[QuerySet, dict[str, int]]
        ] = DeviceCamera.objects.filter(
            is_deleted=False, cube_id=device_cube_id
        ).values(
            "id"
        )
        for valid_device_camera in valid_device_camera_list:
            cls.connect_status_dict[device_cube_id][
                valid_device_camera["id"]
            ].set_connect_status(connect_status)

    @classmethod
    def set_connect_status_dict_from_device_camera_id_list(
        cls,
        device_cube_id: int,
        device_camera_id_list: list[int],
        connect_status: ConnectStatus = ConnectStatus(
            modified_time=None, is_connected=None
        ),
    ) -> None:
        """
        It is useful when Django web server can not connect the camera from ml_model Cube program

        + Warning
            - following items must be verified before using this function
                - ObjectDoesNotExist Exception from device_cube
                - device_cube.is_deleted False
                - device_cube_id is same for all device_camera_id_list
        """
        for device_camera_id in device_camera_id_list:
            cls.connect_status_dict[device_cube_id][
                device_camera_id
            ].set_connect_status(connect_status)

    @classmethod
    def set_connect_status_dict_from_connect_threshold_seconds(
        cls, connect_threshold_seconds: ConnectStatus
    ) -> None:
        """
        It is useful when Django web server checks connect status at regular intervals.
        """
        valid_device_camera_list: list[Union[QuerySet, dict[str, int]]] = (
            DeviceCamera.objects.select_related("cube")
            .filter(is_deleted=False, cube__is_deleted=False)
            .values("id", "cube_id")
        )

        for valid_device_camera in valid_device_camera_list:
            cls.connect_status_dict["cube_id"][
                valid_device_camera["id"]
            ].set_connect_status_from_connect_threshold_seconds(
                connect_threshold_seconds
            )

    @classmethod
    def add_cube_id_key_list_of_connect_status_dict(
        cls, device_cube_id_key_list: list[int]
    ) -> None:
        for cube_id_key in device_cube_id_key_list:
            cls.connect_status_dict[cube_id_key] = {}

    @classmethod
    def delete_cube_id_key_list_of_connect_status_dict(
        cls, device_cube_id_key_list: list[int]
    ) -> None:
        for cube_id_key in device_cube_id_key_list:
            try:
                del cls.connect_status_dict[cube_id_key]
            except KeyError as e:
                pass
        

    @classmethod
    def add_camera_id_key_list_of_connect_status_dict(
        cls, device_cube_id: int, device_camera_id_key_list: list[int]
    ) -> None:
        for camera_id_key in device_camera_id_key_list:
            cls.connect_status_dict[device_cube_id][camera_id_key] = ConnectStatus()

    @classmethod
    def delete_camera_id_key_list_of_connect_status_dict(
        cls, device_cube_id: int, device_camera_id_key_list: list[int]
    ) -> None:
        for camera_id_key in device_camera_id_key_list:
            try:
                del cls.connect_status_dict[device_cube_id][camera_id_key]
            except KeyError as e:
                pass
            
    @classmethod
    def set_most_recent_device_camera_event_dict(
        cls, device_camera_event_list: list[DeviceCameraEvent]
    ) -> None:
        for device_camera_event in device_camera_event_list:
            if device_camera_event.image:
                cls.most_recent_device_camera_event_image_dict[
                    device_camera_event.camera_id
                ] = device_camera_event.image
            else:
                cls.most_recent_device_camera_event_image_dict[
                    device_camera_event.camera_id
                ] = GlobalConfig.DEVICE_CAMERA_EVENT_NONE_IMAGE

    @classmethod
    def set_last_connect_status_dict_to_database(cls) -> None:
        """It can be used like an in-memory DB."""
        DeviceCameraConnectStatus.set_last_connect_status_dict_to_database(
            cls.connect_status_dict
        )


class DummyDictDataFromMqtt(TypedDict):
    cube_serial_number: int
    camera_ip_address: int
    event_type_id: int
    event_created_timestamp: float
    base64_encoded_image: str


@dataclasses.dataclass
class DataFromMqtt:
    cube_serial_number: int = dataclasses.field(default_factory=int)
    camera_ip_address: int = dataclasses.field(default_factory=int)
    event_type_id: int = dataclasses.field(default_factory=int)
    event_created_timestamp: float = dataclasses.field(default_factory=float)
    base64_encoded_image: str = dataclasses.field(default_factory=str)

    @staticmethod
    def convert_to_class_instance_from_dict(
        dummy_dict_data_from_mqtt: DummyDictDataFromMqtt,
    ) -> DataFromMqtt:
        """
        + Note
            - If original data is Django "request" instasnce, instead following example statement:
                requested_data = SimpleNamespace(**request.POST.dict())
        """
        return DummyDictDataFromMqtt(
            cube_serial_number=dummy_dict_data_from_mqtt["cube_serial_number"],
            camera_ip_address=dummy_dict_data_from_mqtt["camera_ip_address"],
            event_type_id=dummy_dict_data_from_mqtt["event_type_id"],
            event_created_timestamp=dummy_dict_data_from_mqtt[
                "event_created_timestamp"
            ],
            base64_image=dummy_dict_data_from_mqtt["base64_image"],
        )

    @staticmethod
    def main_process(data_list_from_mqtt: list[DataFromMqtt]) -> None:
        """entry point of process when data comes from MQTT.

        TO DO: image path set, classify event

        + Note
            - when real camera ip_address is changed, it may occurs some errors.
            - it assumes cube_id of data of one request is same.
            - it save device_camera_event with event type of a camera error also when ml_model cube cannot connect to the camera.

        """
        if not data_list_from_mqtt:
            return

        try:
            # preprocess
            device_cube: Union[QuerySet, DeviceCube] = DeviceCube.objects.filter(
                serial_number=data_list_from_mqtt[0].cube_serial_number
            ).first()
            if device_cube.is_deleted:
                return

            valid_device_camera_list: list[
                Union[QuerySet, DeviceCamera]
            ] = DeviceCamera.objects.filter(
                cube_id=device_cube.id,
                is_deleted=False,
                ip_address__in=[x.camera_ip_address for x in data_list_from_mqtt],
            )

            camera_id_to_data_list_from_mqtt: dict[int, DataFromMqtt] = {}
            for valid_device_camera in valid_device_camera_list:
                for data_from_mqtt in data_list_from_mqtt:
                    if (
                        data_from_mqtt.camera_ip_address
                        == valid_device_camera.ip_address
                    ):
                        camera_id_to_data_list_from_mqtt[
                            valid_device_camera.id
                        ] = data_from_mqtt
                        break

            threat_event_list: list[
                Union[QuerySet, ThreatEvent]
            ] = ThreatEvent.objects.all()
            threat_event_id_to_threat_event: dict[int, Union[QuerySet, ThreatEvent]] = {
                x.id: x for x in threat_event_list
            }

            device_camera_event_list: list[DeviceCameraEvent] = []
            for valid_device_camera in valid_device_camera_list:
                matched_data_from_mqtt: DataFromMqtt = camera_id_to_data_list_from_mqtt[
                    valid_device_camera.id
                ]
                base64_decoded_image: bytes = base64.b64decode(
                    matched_data_from_mqtt.base64_encoded_image
                )
                base64_image_extension: str = imghdr.what(None, h=base64_decoded_image)
                current_datetime_str: str = datetime.datetime.now(
                    pytz.timezone("Asia/Seoul")
                ).strftime("%Y%m%d_%H%M%S")

                image_in_django: ContentFile = ContentFile(
                    base64_decoded_image,
                    name=f"{GlobalConfig.DEVICE_CAMERA_EVENT_NAME_PREFIX}{current_datetime_str}.{base64_image_extension}",
                )

                device_camera_event_list.append(
                    DeviceCameraEvent(
                        camera=valid_device_camera,
                        threat_event=threat_event_id_to_threat_event[
                            matched_data_from_mqtt.event_type_id
                        ],
                        image=image_in_django,
                        created_timestamp=matched_data_from_mqtt.event_created_timestamp,
                    )
                )
            device_camera_error_event_list: list[DeviceCameraEvent] = []
            device_camera_valid_threat_event_list: list[DeviceCameraEvent] = []
            device_camera_real_threat_event_list: list[DeviceCameraEvent] = []
            for device_camera_event in device_camera_event_list:
                if (
                    device_camera_event.threat_event.id
                    == GlobalConfig.DEVICE_CAMERA_EVENT_INDEX_THAT_IS_CAMERA_ERROR
                ):
                    device_camera_error_event_list.append(device_camera_event)
                else:
                    device_camera_valid_threat_event_list.append(device_camera_event)

                if device_camera_event.threat_event.id not in [
                    GlobalConfig.DEVICE_CAMERA_EVENT_INDEX_THAT_IS_NORMAL,
                    GlobalConfig.DEVICE_CAMERA_EVENT_INDEX_THAT_IS_CAMERA_ERROR,
                ]:
                    device_camera_real_threat_event_list.append(device_camera_event)

            # process
            with transaction.atomic():
                DeviceCameraEvent.objects.bulk_create(device_camera_event_list)
            GlobalDeivceCamera.set_most_recent_device_camera_event_dict(
                device_camera_event_list
            )
            GlobalDeivceCamera.set_connect_status_dict_from_device_camera_id_list(
                device_cube_id=device_cube.id,
                device_camera_id_list=[
                    x.camera.id for x in device_camera_error_event_list
                ],
            )
            GlobalDeivceCamera.set_connect_status_dict_from_device_camera_id_list(
                device_cube_id=device_cube.id,
                device_camera_id_list=[
                    x.camera.id for x in device_camera_valid_threat_event_list
                ],
                connect_status=ConnectStatus(
                    modified_time=time.time(), is_connected=True
                ),
            )

            # test
            # base64_encoded_image: str = "R0lGODlhDwAPAKECAAAAzMzM/////wAAACwAAAAADwAPAAACIISPeQHsrZ5ModrLlN48CXF8m2iQ3YmmKqVlRtW4MLwWACH+H09wdGltaXplZCBieSBVbGVhZCBTbWFydFNhdmVyIQAAOw=="
        except ObjectDoesNotExist as e:
            logging.error(f"fail for saving device camera events.: {e}")
        except KeyError as e:
            logging.error(f"fail for saving device camera events.: {e}")
        except Exception as e:
            logging.error(f"fail for saving device camera events.: {e}")
        else:
            logging.info("success for saving device camera events.")
            # pass serializers to frontend
            # alert for device_camera_real_threat_event_list
