"""TODO: Delete this file. instead use serializers/device_camera
"""
from django.contrib.auth.models import User, AnonymousUser
from django.core.exceptions import PermissionDenied
from django.db.models.aggregates import Count
from django.db.models.expressions import F
from django.db.models.query import QuerySet
from django.db.models.query_utils import Q
from django.http import Http404
from django.http.request import HttpRequest
from io import BytesIO
from ml_model.global_object import GlobalConfig, GlobalDeivceCamera, GlobalResponseMessage
from ml_model.model.device_camera import DeviceCamera, DeviceCameraEvent
from ml_model.model.device_cube import DeviceCube
from ml_model.serializers.device_camera import DeviceCameraSerializer, DeviceCameraUpdateTypedDict
from ml_model.utilities.utils_iterable import (
    bulk_update_objects_from_dicts,
    validate_iterable_comply_optional_typed_dict,
)
from rest_framework import viewsets, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, action
from rest_framework.generics import get_object_or_404
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
import itertools
import logging
import inspect


logger = logging.getLogger(__name__)

class DeviceCameraViewSet(viewsets.ModelViewSet):
    authentication_classes = [TokenAuthentication]
    serializer_class = DeviceCameraSerializer

    def get_queryset(self) -> Union[QuerySet, list[DeviceCamera]]:
        if isinstance(self.request.user, AnonymousUser):
            raise PermissionDenied
        else:
            return (
                DeviceCamera.objects.select_related("cube")
                .filter(is_deleted=False, cube__user_id=self.request.user.id)
                .all()
            )

    def list(self, request: Request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        if "page" in request.query_params:
            page = self.paginate_queryset(queryset)
            if page is not None:
                serializer: DeviceCameraSerializer = self.get_serializer(
                    page, many=True
                )
                return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request: Request) -> Response:
        # preprocess
        request_data: list[dict] = None
        if isinstance(request.data, dict):
            request_data = [request.data]
        elif isinstance(request.data, list):
            if len(request.data) < 1:
                return Response(status=status.HTTP_204_NO_CONTENT)
            request_data = request.data

        user_id: int = (
            User.objects.filter(username=request.user)
            .values_list("id", flat=True)
            .first()
        )

        cube_in_db: Union[QuerySet, DeviceCube] = (
            DeviceCube.objects.filter(id=request_data[0]["cube_id"])
            .values("id", "user_id")
            .first()
        )
        if user_id != cube_in_db["user_id"]:
            raise PermissionDenied

        for one_request_data in request_data:
            one_request_data["cube_id"] = cube_in_db["id"]

        serializer: DeviceCameraSerializer = self.get_serializer(
            data=request_data, many=True
        )
        serializer.is_valid(raise_exception=True)

        # process
        self.perform_create(serializer)
        GlobalDeivceCamera.add_camera_id_key_list_of_connect_status_dict(
            device_cube_id=serializer.data[0]["cube_id"],
            device_camera_id_key_list=[x["id"] for x in serializer.data],
        )

        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

    def update(self, request: Request, *args, **kwargs):
        raise Http404

    def partial_update(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        if "cube_id" in request.data.keys():
            user_id: int = request.user.id
            cube_in_db: Union[QuerySet, DeviceCube] = (
                DeviceCube.objects.filter(id=request.data["cube_id"])
                .values("id", "user_id")
                .first()
            )
            if user_id != cube_in_db["user_id"]:
                raise PermissionDenied

        instance: DeviceCamera = self.get_object()
        serializer: DeviceCameraSerializer = self.get_serializer(
            instance, data=request.data, partial=True
        )
        serializer.is_valid(raise_exception=True)

        # process
        self.perform_update(serializer)
        if getattr(instance, "_prefetched_objects_cache", None):
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    @action(detail=False, methods=["patch"], url_path="in-batch")
    def partial_update_in_batch(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        request_user: User = request.user
        if isinstance(request.user, AnonymousUser):
            raise PermissionDenied
        request_data_list: list[dict] = None
        if isinstance(request.data, dict):
            request_data_list = [request.data]
        elif isinstance(request.data, list):
            if len(request.data) < 1:
                return Response(status=status.HTTP_204_NO_CONTENT)
            request_data_list = request.data

        required_key_list: list[str] = ["id"]
        fields_to_be_updated: list[str] = [key for key in DeviceCameraUpdateTypedDict.__annotations__.keys() if key not in required_key_list]

        if not validate_iterable_comply_optional_typed_dict(
            targets_iterable=request_data_list,
            typed_dict_class=DeviceCameraUpdateTypedDict,
            required_key_list=required_key_list
        ):
            return Response(
                GlobalResponseMessage.ARGUMENTS_ERROR_TO_SET_DEVICE_CUBE,
                status=status.HTTP_400_BAD_REQUEST,
            )

        request_data_device_camera_id_list: list[int] = [one_request_data["id"] for one_request_data in request_data_list]

        device_camera_list_in_db: list[
            Union[QuerySet, DeviceCamera]
        ] = DeviceCamera.objects.select_related("cube").filter(id__in=request_data_device_camera_id_list)

        device_camera_owner: set = set(
            [device_camera_in_db.cube.user_id for device_camera_in_db in device_camera_list_in_db]
        )
        if len(device_camera_owner) > 1 or not request_user.id in device_camera_owner:
            raise PermissionDenied

        # process
        updated_objects = bulk_update_objects_from_dicts(
            objects=device_camera_list_in_db,
            dicts=request_data_list,
            key_or_attribute_name="id",
            fields_to_be_updated=fields_to_be_updated,
        )
        DeviceCamera.objects.bulk_update(
            objs=updated_objects, fields=fields_to_be_updated
        )

        response_serializer: DeviceCameraSerializer = DeviceCameraSerializer(
            updated_objects, many=True
        )

        return Response(response_serializer.data)

    def destroy(self, request: Request, *args, **kwargs) -> Response:
        instance: DeviceCamera = self.get_object()
        instance.is_deleted = True
        instance.save()
        GlobalDeivceCamera.delete_camera_id_key_list_of_connect_status_dict(
            device_cube_id=instance.cube_id,
            device_camera_id_key_list=[instance.id],
        )
        return Response(status=status.HTTP_204_NO_CONTENT)

    @partial_update_in_batch.mapping.delete
    def destroy_in_batch(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        request_user: User = request.user
        if isinstance(request.user, AnonymousUser):
            raise PermissionDenied
        if not isinstance(request.data, list):
            return Http404
        if len(request.data) < 1:
            return Http404
        request_data_device_camera_id_list: list[int] = request.data

        device_camera_list_in_db: list[dict[str, int]] = list(
            (
                DeviceCamera.objects.select_related("cube")
                .filter(id__in=request_data_device_camera_id_list)
                .annotate(user_id=F("cube__user_id"))
                .values("id", "cube_id", "user_id")
                .order_by("cube_id")
            )
        )
        user_id_in_db: set[int] = set([x["user_id"] for x in device_camera_list_in_db])

        # process
        if (request_user.id not in user_id_in_db) or (len(user_id_in_db) > 1):
            raise PermissionDenied
        DeviceCamera.objects.filter(id__in=request_data_device_camera_id_list).update(
            is_deleted=True
        )
        for device_camera_id_group in [
            list(g)
            for k, g in itertools.groupby(
                device_camera_list_in_db,
                lambda item: (item["cube_id"]),
            )
        ]:
            GlobalDeivceCamera.delete_camera_id_key_list_of_connect_status_dict(
                device_cube_id=device_camera_id_group[0]["cube_id"],
                device_camera_id_key_list=[x["id"] for x in device_camera_id_group],
            )

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=["get"], url_path="count")
    def count(self, request: Request, *args, **kwargs) -> Response:
        return Response(
            {"count": self.get_queryset().count()}, status=status.HTTP_200_OK
        )
