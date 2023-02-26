import itertools
from django.contrib.auth.models import User
from django.db.models.expressions import F
from django.db.models.query import QuerySet
from django.db.models.query_utils import Q
from django.http import Http404
from django.core.exceptions import PermissionDenied
from io import BytesIO
from ml_model.global_object import GlobalConfig, GlobalDeivceCamera, GlobalResponseMessage
from ml_model.model.device_camera import DeviceCamera, DeviceCameraEvent
from ml_model.model.device_cube import DeviceCube
from rest_framework import serializers, viewsets, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import action
from rest_framework.generics import get_object_or_404
from rest_framework.parsers import JSONParser
from rest_framework.permissions import (
    IsAuthenticated,
    IsAuthenticatedOrReadOnly,
    BasePermission,
    SAFE_METHODS,
)
from rest_framework.request import Request
from rest_framework.response import Response
from typing import Optional, Union
import dataclasses
from typing import TypedDict


class DeviceCameraUpdateTypedDict(TypedDict):
    id: int
    ip_address: str
    device_id: str
    device_pw: str
    location: str
    cube_id: int

class DeviceCameraSerializer(serializers.ModelSerializer):
    """It is used to CRUD in page of device_camera."""

    is_connected = serializers.SerializerMethodField(read_only=True)
    cube = serializers.SlugRelatedField(slug_field="serial_number", read_only=True)
    cube_id = serializers.PrimaryKeyRelatedField(
        source="cube",
        queryset=DeviceCube.objects.filter(is_deleted=False).all(),
    )

    most_recent_event_image = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = DeviceCamera
        fields = [
            "id",
            "cube",
            "cube_id",
            "ip_address",
            "device_id",
            "device_pw",
            "location",
            "is_connected",
            "most_recent_event_image",
        ]

    def get_is_connected(self, device_camera: DeviceCamera):
        try:
            device_camera.connect_status = GlobalDeivceCamera.connect_status_dict[
                device_camera.cube_id
            ][device_camera.id]
        except Exception:
            return False
        else:
            return device_camera.connect_status.is_connected

    def get_most_recent_event_image(self, device_camera: DeviceCamera):
        try:
            most_recent_event_image: str = (
                GlobalDeivceCamera.most_recent_device_camera_event_image_dict[
                    device_camera.id
                ]
            )
        except Exception:
            return None
        else:
            return most_recent_event_image


class DeviceCameraConnectStatusSerializer(serializers.ModelSerializer):
    """It is used to pass notifications for connect status of device_camera to each user."""

    is_connected = serializers.SerializerMethodField()

    class Meta:
        model = DeviceCamera
        fields = ["id", "is_connected"]
        read_only_fields = ["id", "is_connected"]

    def get_is_connected(self, device_camera: DeviceCamera):
        try:
            device_camera.connect_status = GlobalDeivceCamera.connect_status_dict[
                device_camera.cube_id
            ][device_camera.id]
        except Exception:
            return None
        else:
            return device_camera.connect_status.is_connected


class DeviceCameraEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceCameraEvent
        fields = [
            "id",
            "threat_event__id",
            "device_camera__location",
            "image",
            "created_timestamp",
            "cube__serial_number",
        ]

