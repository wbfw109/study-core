from django.contrib.auth.models import User, AnonymousUser
from django.core.exceptions import PermissionDenied
from django.db.models.query import QuerySet
from django.db.models.query_utils import Q
from django.http import Http404
from ml_model.global_object import GlobalDeivceCamera, GlobalResponseMessage
from ml_model.model.device_cube import DeviceCube
from ml_model.serializers.device_cube import (
    DeviceCubeRegisterSerializer,
    DeviceCubeSerializer,
    DeviceCubeSetFirmwareSerializer,
)
from ml_model.utilities.utils_iterable import bulk_update_objects_from_dicts
from rest_framework import viewsets, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import action, api_view
from rest_framework.request import Request
from rest_framework.response import Response
from typing import Union
import logging
import random
import string

logger = logging.getLogger(__name__)


class DeviceCubeViewSet(viewsets.ModelViewSet):
    authentication_classes = [TokenAuthentication]
    serializer_class = DeviceCubeSerializer

    def get_queryset(self, filters: list[Q] = []) -> Union[QuerySet, list[DeviceCube]]:
        if isinstance(self.request.user, AnonymousUser):
            raise PermissionDenied
        else:
            return DeviceCube.objects.filter(
                *filters,
                is_deleted=False,
                user_id=self.request.user.id,
            ).all()

    def list(self, request: Request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        if "page" in request.query_params:
            page = self.paginate_queryset(queryset)
            if page is not None:
                serializer: DeviceCubeSerializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request: Request) -> Response:
        return Http404

    def update(self, request: Request, *args, **kwargs):
        return Http404

    def partial_update(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        request_user: User = request.user

        request_data_keys = request.data.keys()

        if "serial_number" not in request_data_keys:
            return Response(
                {"error": GlobalResponseMessage.ARGUMENTS_ERROR_TO_VALIDATE_SERIAL_NUMBER},
                status=status.HTTP_404_NOT_FOUND,
            )

        request_serial_number: str = request.data["serial_number"]

        instance: DeviceCube = self.get_object()
        if not (instance.user_id == None or instance.user_id == request_user.id):
            raise PermissionDenied
        if instance.serial_number != request_serial_number:
            raise PermissionDenied

        serializer: DeviceCubeSerializer = self.get_serializer(
            instance, data=request.data, partial=True
        )
        serializer.is_valid(raise_exception=True)

        # process
        self.perform_update(serializer)
        if getattr(instance, "_prefetched_objects_cache", None):
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    @action(detail=False, methods=["patch"], url_path="register/in-batch")
    def register_in_batch(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        request_user: User = request.user
        if isinstance(request.user, AnonymousUser):
            raise PermissionDenied
        request_data: list[dict] = None
        if isinstance(request.data, dict):
            request_data = [request.data]
        elif isinstance(request.data, list):
            if len(request.data) < 1:
                return Response(status=status.HTTP_204_NO_CONTENT)
            request_data = request.data

        request_data_device_cube_serial_number_list: list[int] = []
        for one_request_data in request_data:
            one_request_data_keys = one_request_data.keys()
            if not {"serial_number"}.issubset(one_request_data_keys) or (
                not isinstance(one_request_data["serial_number"], str)
            ):
                return Response(
                    GlobalResponseMessage.ARGUMENTS_ERROR_TO_REGISTER_DEVICE_CUBE,
                    status=status.HTTP_400_BAD_REQUEST,
                )
            request_data_device_cube_serial_number_list.append(
                one_request_data["serial_number"]
            )

        device_cube_list_in_db: list[
            Union[QuerySet, DeviceCube]
        ] = DeviceCube.objects.filter(
            serial_number__in=request_data_device_cube_serial_number_list,
            is_deleted=False,
            user_id=None,
        )
        if len(device_cube_list_in_db) != len(request_data):
            return Response(
                GlobalResponseMessage.UNKNOWN_SERIAL_NUMBER,
                status=status.HTTP_400_BAD_REQUEST,
            )

        # process
        for device_cube_in_db in device_cube_list_in_db:
            device_cube_in_db.user_id = request_user.id
        fields_to_be_updated = ["user_id"]

        DeviceCube.objects.bulk_update(
            objs=device_cube_list_in_db, fields=fields_to_be_updated
        )

        response_serializer: DeviceCubeRegisterSerializer = (
            DeviceCubeRegisterSerializer(device_cube_list_in_db, many=True)
        )

        return Response(response_serializer.data)

    @action(detail=False, methods=["patch"], url_path="firmware/in-batch")
    def set_firmware_in_batch(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        request_user: User = request.user
        if isinstance(request.user, AnonymousUser):
            raise PermissionDenied
        if not request_user.is_superuser:
            raise PermissionDenied

        request_data_list: list[dict] = None
        if isinstance(request.data, dict):
            request_data_list = [request.data]
        elif isinstance(request.data, list):
            if len(request.data) < 1:
                return Response(status=status.HTTP_204_NO_CONTENT)
            request_data_list = request.data
        request_data_device_cube_id_list: list[int] = []
        for one_request_data in request_data_list:
            one_request_data_keys = one_request_data.keys()
            if not {"id", "firmware"}.issubset(one_request_data_keys) or (
                not (
                    isinstance(one_request_data["id"], int)
                    and isinstance(one_request_data["firmware"], str)
                )
            ):
                return Response(
                    GlobalResponseMessage.ARGUMENTS_ERROR_SET_DEVICE_CUBE_FIRMWARE,
                    status=status.HTTP_400_BAD_REQUEST,
                )
            request_data_device_cube_id_list.append(one_request_data["id"])

        device_cube_list_in_db: list[
            Union[QuerySet, DeviceCube]
        ] = DeviceCube.objects.filter(id__in=request_data_device_cube_id_list, is_deleted=False)

        # process
        fields_to_be_updated = ["firmware"]
        updated_objects = bulk_update_objects_from_dicts(
            objects=device_cube_list_in_db,
            dicts=request_data_list,
            key_or_attribute_name="id",
            fields_to_be_updated=fields_to_be_updated,
        )
        DeviceCube.objects.bulk_update(
            objs=updated_objects, fields=fields_to_be_updated
        )

        response_serializer: DeviceCubeSetFirmwareSerializer = (
            DeviceCubeSetFirmwareSerializer(updated_objects, many=True)
        )

        return Response(response_serializer.data)

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
        request_data_device_cube_id_list: list[int] = []
        for one_request_data in request_data_list:
            one_request_data_keys = one_request_data.keys()
            if not {"id", "name"}.issubset(one_request_data_keys) or (
                not (
                    isinstance(one_request_data["id"], int)
                    and isinstance(one_request_data["name"], str)
                )
            ):
                return Response(
                    GlobalResponseMessage.ARGUMENTS_ERROR_TO_SET_DEVICE_CUBE_NAME,
                    status=status.HTTP_400_BAD_REQUEST,
                )
            request_data_device_cube_id_list.append(one_request_data["id"])

        device_cube_list_in_db: list[
            Union[QuerySet, DeviceCube]
        ] = DeviceCube.objects.filter(id__in=request_data_device_cube_id_list, is_deleted=False)
        
        device_cube_owner: set = set([device_cube_in_db.user_id for device_cube_in_db in device_cube_list_in_db])
        if len(device_cube_owner) > 1 or request_user.id not in device_cube_owner:
            raise PermissionDenied


        # process
        fields_to_be_updated = ["name"]
        updated_objects = bulk_update_objects_from_dicts(
            objects=device_cube_list_in_db,
            dicts=request_data_list,
            key_or_attribute_name="id",
            fields_to_be_updated=fields_to_be_updated,
        )
        DeviceCube.objects.bulk_update(
            objs=updated_objects, fields=fields_to_be_updated
        )
        GlobalDeivceCamera.add_cube_id_key_list_of_connect_status_dict(
            device_cube_id_key_list=[x.id for x in updated_objects],
        )

        response_serializer: DeviceCubeSerializer = (
            DeviceCubeSerializer(updated_objects, many=True)
        )

        return Response(response_serializer.data)

    def destroy(self, request: Request, *args, **kwargs) -> Response:
        instance: DeviceCube = self.get_object()
        instance.is_deleted = True
        instance.save()
        GlobalDeivceCamera.delete_cube_id_key_list_of_connect_status_dict(
            device_cube_id_key_list=[instance.id],
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
        request_data_device_cube_id_list: list[int] = request.data

        device_cube_list_in_db: list[dict[str, int]] = list(
            (
                DeviceCube.objects.filter(
                    id__in=request_data_device_cube_id_list
                ).values("id", "user_id")
                
            )
        )
        device_cube_user_id_list_in_db: set[int] = set(
            [x["user_id"] for x in device_cube_list_in_db]
        )

        # process
        if (request_user.id not in device_cube_user_id_list_in_db) or (
            len(device_cube_user_id_list_in_db) > 1
        ):
            raise PermissionDenied

        DeviceCube.objects.filter(id__in=request_data_device_cube_id_list).update(
            is_deleted=True
        )
        GlobalDeivceCamera.delete_cube_id_key_list_of_connect_status_dict(
            device_cube_id_key_list=[x["id"] for x in device_cube_list_in_db],
        )

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=["get"], url_path="count")
    def count(self, request: Request, *args, **kwargs) -> Response:
        return Response(
            {"count": self.get_queryset().count()}, status=status.HTTP_200_OK
        )


@api_view(["GET"])
def create_cube(request):
    firmware = request.GET.get("firmware", None)
    count = request.GET.get("count", None)
    if firmware is None or count is None:
        return Response("please input <firmware, count> params")

    check_dic = DeviceCube.objects.all().values("serial_number")
    check_list = []
    for value in check_dic:
        check_list.append(value["serial_number"])

    serial_list = set()
    while True:
        serial = create_serial()

        if serial in check_list:
            continue
        serial_list.add(serial)

        if len(serial_list) == int(count):
            break
    serial_list = list(serial_list)

    for i in range(int(count)):
        cube = DeviceCube(serial_number=serial_list[i], firmware=firmware)
        cube.save()

    return Response("create success")


def create_serial():
    str_upper = string.ascii_uppercase
    str_digits = string.digits
    serial = ""

    for i in range(1, 17):
        ran = random.randint(0, 1)
        if ran == 0:
            serial += random.choice(str_upper)
        else:
            serial += random.choice(str_digits)

        if i % 4 == 0:
            serial += "-"
    return serial[:-1]


# @api_view(["GET", "POST"])
# def cube_list(request):
#     if request.method == "GET":
#         logger.debug("get request about cube list")

#         user_name = request.GET["user_name"]
#         user_id = User.objects.filter(username=user_name).values("id")[0]["id"]

#         start = int(request.GET["start"])
#         end = int(request.GET["end"])

#         # user id 에 해당하는 큐브 값 갖고와야됨
#         total = DeviceCube.objects.filter(is_deleted=False, user_id=user_id).aggregate(
#             Count("id")
#         )
#         if start < 0 or start >= total["id__count"]:
#             # if start < 0 or start >= total['id__count']:
#             return Response([[], 0])
#         if total["id__count"] < end:
#             end = total["id__count"]
#         data = DeviceCube.objects.filter(is_deleted=False, user_id=user_id)[start:end]

#         logger.debug(len(data))

#         serializers = DeviceCubeSerializer(
#             data, context={"request": request}, many=True
#         )
#         data_part = serializers.data

#         return Response([data_part, total["id__count"]])
#     elif request.method == "POST":
#         logger.debug("insert/update/delete cube")

#         data = request.data
#         method_type = data["type"]
#         serial = data["serial"]

#         try:
#             if method_type == "create":
#                 user_name = request.data["user_name"]
#                 user_id = User.objects.filter(username=user_name).values("id")[0]["id"]
#                 # serial number 맞는지 & 이미 등록된 시리얼인지 확인
#                 get_cube_list = DeviceCube.objects.filter(
#                     serial_number=serial, user_id=None
#                 ).values()
#                 if 1 != len(get_cube_list):
#                     return Response({"ok": False, "why": "serial"})

#                 name = data["name"]

#                 cube = DeviceCube.objects.get(serial_number=serial)
#                 cube.name = name
#                 cube.user_id = User.objects.get(id=user_id)
#                 cube.is_deleted = False
#                 cube.save()
#             elif method_type == "modify":
#                 name = data["name"]

#                 cube = DeviceCube.objects.get(serial_number=serial[0])
#                 cube.name = name
#                 cube.save()
#             elif method_type == "delete":
#                 for serial_number in serial:
#                     cube = DeviceCube.objects.get(serial_number=serial_number)
#                     cube.name = ""
#                     cube.user_id = None
#                     cube.is_deleted = True
#                     cube.save()
#         except Exception as ex:
#             return Response({"ok": False, "why": ex})
#         else:
#             return Response({"ok": method_type})


# @api_view(["GET"])
# def get_firmware(request):
#     serial = request.GET.get("serial[]", None)
#     cube = DeviceCube.objects.filter(serial_number=serial).values("firmware")[0]

#     return Response({"firmware": cube["firmware"]})
