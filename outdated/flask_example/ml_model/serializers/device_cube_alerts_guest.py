import itertools
from typing import Optional, Union

from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.db.models.expressions import F
from django.db.models.query import QuerySet
from django.db.models.query_utils import Q
from django.http import Http404
from rest_framework import serializers, status, viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response

from ml_model.global_object import GlobalDeivceCamera
from ml_model.model.device_cube import DeviceCube, DeviceCubeAlertsGuest, Guest
from ml_model.serializers.device_cube import DeviceCubeMinimalSerializer

"""
연락처
id	int Auto Increment
phone_number	varchar(15)	
created_datetime	datetime(6)	
cube_id	int	

프론트 페이지 요청
    delete cube_id_list from guest: phone number, cube_id_list
        => if guest is none, delete guest
    add cube_id_list from guest: phone number, cube_id_list
        => if guest ie none, create gues

    권한 검증
        => if cube_id_list 의 user_id_list
            pass

보여주는 것: cube serial number, cube_name


하나의 phone number 는 여러 개의 큐브에 종속 될 수 있다.
하나의 큐브는 여러 개의 phone nubmer 를 보유할 수 있다.
phone number 변경.
Many To Many  단방향
    수신자가 중심.

연락처가 일치? 없으면 생성해야 함.
unique
device_cube_alerts_guests 가 없으면 생성해야 함.

쿼리가 기존 연락처에서 이름변경+큐브 연관 레코드 추가까지 같이 될 수도 있음.

-> 어차피 guest_id, device_cube_id_list 에 접근권한 있는지 검증해야 함.
    

동일한 phone number 와 동일한 cube_id를 참조하고 있는 중복 로우 생성 불가.
Guest 는 is_deleted 필요? 그냥 삭제?
"""


class DeviceCubeAlertsGuestSerializer(serializers.ModelSerializer):
    """It is used to CRUD in page of device_camera."""

    device_cubes = DeviceCubeMinimalSerializer(many=True)

    class Meta:
        model = DeviceCubeAlertsGuest
        fields = ["id", "device_cubes"]

    def create(self, validated_data: dict):
        tracks_data = validated_data.pop("tracks")
        album = DeviceCubeAlertsGuest.objects.create(**validated_data)
        for track_data in tracks_data:
            Track.objects.create(album=album, **track_data)
        return album


class DeviceCubeAlertsGuestOperationGetSerializer(serializers.ModelSerializer):
    """
    GET method structure
        [
            {
                id: guest.id
                phone_number: guest.phone_number
                device_cube_alerts_guests: [
                    {
                        id: device_cube_alerts_guest.id,
                        device_cubes: {
                            id: device_cube.id,
                            name: device_cube.name
                            serial_number: device_cube.serial_number
                        }
                    }
                ]
            }
        ]

    verify permissions related with user_id:
        - DeviceCubeAlertsGuest.objects.filter(device_cube__user_id=user_id)
    """

    device_cube_alerts_guests = DeviceCubeAlertsGuestSerializer(many=True)

    class Meta:
        model = Guest
        fields = [
            "id",
            "phone_number",
            "device_cube_alerts_guests",
        ]

    def create(self, validated_data: dict):
        device_cube_alerts_guests_data: dict = validated_data.pop(
            "device_cube_alerts_guests"
        )
        device_cube_alerts_guests_data["guest_id"] = validated_data["id"]
        self.device_cube_alerts_guests

        device_cube_alert_guests_list: list[
            DeviceCubeAlertsGuest
        ] = DeviceCubeAlertsGuest.objects.create(**validated_data)
        for track_data in tracks_data:
            Track.objects.create(album=device_cube_alert_guest_list, **track_data)
        return device_cube_alert_guest_list


class DeviceCubeAlertsGuestOperationDeleteSerializer(serializers.ModelSerializer):
    """
    DELETE method structure
        [device_cube_alerts_guest.id]

    verify permissions related with user_id:
        - DeviceCubeAlertsGuest.objects.select_related("device_cube").filter(
            id__in=[device_cube_alerts_guest.id]
        ).values_list("device_cube__user_id", flat=True).distinct()
    """

    class Meta:
        model = Guest
        fields = [
            "id",
            "phone_number",
            "device_cube_alerts_guests",
        ]


class DeviceCubeAlertsGuestOperationPostSerializer(serializers.ModelSerializer):
    """
    POST method structure
        [
            {
                phone_number: guest.phone_number
                device_cubes: {
                    id: device_cube.id
                }
            }
        ]

    verify permissions related with user_id:
        - DeviceCube.objects.select_related("device_cube").filter(
            id__in=[device_cubes.id]
        ).values_list("user_id", flat=True)
    """

    class Meta:
        model = Guest
        fields = [
            "id",
            "phone_number",
            "device_cube_alerts_guests",
        ]


class DeviceCubeAlertsGuestOperationPatchSerializer(serializers.ModelSerializer):
    """
    PATCH method structure (allowed optional)
        [
            {
                id: guest.id
                phone_number: guest.phone_number
                device_cubes_to_be_added: [
                    {
                        id: cube.id
                    }
                ]
                device_cubes_to_be_deleted: [
                    {
                        id: cube.id
                    }
                ]
            }
        ]

    verify permissions related with user_id:
        - Guest.objects.filter(
            id__in=[guest.id]
        ).values_list("user_id", flat=True).distinct()
        - DeviceCubeAlertsGuest.objects.select_related("device_cube").filter(
            id__in=[device_cube_alerts_guest.id]
        ).values_list("device_cube__user_id", flat=True).distinct()
    """

    class Meta:
        model = Guest
        fields = [
            "id",
            "phone_number",
            "device_cube_alerts_guests",
        ]


class DeviceCubeAlertsGuestViewSet(viewsets.ModelViewSet):
    authentication_classes = [TokenAuthentication]
    serializer_class: dict[str, serializers.Serializer] = {
        "GET": DeviceCubeAlertsGuestOperationGetSerializer,
        "DELETE": DeviceCubeAlertsGuestOperationDeleteSerializer,
        "POST": DeviceCubeAlertsGuestOperationPostSerializer,
        "PATCH": DeviceCubeAlertsGuestOperationPatchSerializer,
    }

    def get_queryset(self) -> Union[QuerySet, list[Guest]]:
        user_id: int = (
            User.objects.filter(username=self.request.user)
            .values_list("id", flat=True)
            .first()
        )
        return (
            Guest.objects.select_related("cube")
            .filter(is_deleted=False, cube__user_id=user_id)
            .all()
        )

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
            User.objects.filter(username=self.request.user)
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

        serializer = self.get_serializer(data=request_data, many=True)
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
            user_id: int = (
                User.objects.filter(username=self.request.user)
                .values_list("id", flat=True)
                .first()
            )
            cube_in_db: Union[QuerySet, DeviceCube] = (
                DeviceCube.objects.filter(id=request.data["cube_id"])
                .values("id", "user_id")
                .first()
            )
            if user_id != cube_in_db["user_id"]:
                raise PermissionDenied

        instance: Guest = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)

        # process
        self.perform_update(serializer)
        if getattr(instance, "_prefetched_objects_cache", None):
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    @action(detail=False, methods=["patch"], url_name="batch")
    def partial_update_in_batch(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        request_data: list[dict] = None
        if isinstance(request.data, dict):
            request_data = [request.data]
        elif isinstance(request.data, list):
            if len(request.data) < 1:
                return Response(status=status.HTTP_204_NO_CONTENT)
            request_data = request.data

        user_id: int = (
            User.objects.filter(username=self.request.user)
            .values_list("id", flat=True)
            .first()
        )

        request_data_cube_id_list: list[int] = []
        q_of_cube_id_list: Optional[Q] = None
        request_data_camera_id_list: list[int] = []
        q_of_camera_id_list: Optional[Q] = None

        for one_request_data in request_data:
            if "cube_id" in one_request_data.keys():
                request_data_cube_id_list.append(one_request_data["cube_id"])
            request_data_camera_id_list.append(one_request_data["id"])
        if request_data_cube_id_list:
            q_of_cube_id_list = Q(id__in=request_data_cube_id_list)
        if request_data_camera_id_list:
            q_of_camera_id_list = Q(id__in=request_data_camera_id_list)

        user_id_in_db: list[int] = list(
            (
                DeviceCube.objects.filter(q_of_cube_id_list | q_of_camera_id_list)
                .values_list("user_id", flat=True)
                .distinct()
            )
        )
        if (user_id not in user_id_in_db) or (len(user_id_in_db) > 1):
            raise PermissionDenied

        # process
        serializer = self.get_serializer(data=request_data, many=True, partial=True)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        return Response(serializer.data)

    def destroy(self, request: Request, *args, **kwargs) -> Response:
        instance: Guest = self.get_object()
        instance.is_deleted = True
        instance.save()
        GlobalDeivceCamera.delete_camera_id_key_list_of_connect_status_dict(
            device_cube_id=instance.cube_id,
            device_camera_id_key_list=[instance.id],
        )
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=["delete"], url_path="batch")
    def destroy_in_batch(self, request: Request, *args, **kwargs) -> Response:
        # preprocess
        if not isinstance(request.data, list):
            return Http404
        if len(request.data) < 1:
            return Http404
        request_data_camera_id_list: list[int] = request.data

        user_id: int = (
            User.objects.filter(username=self.request.user)
            .values_list("id", flat=True)
            .first()
        )
        device_camera_list_in_db: list[dict[str, int]] = list(
            (
                Guest.objects.select_related("cube")
                .filter(id__in=request_data_camera_id_list)
                .annotate(user_id=F("cube__user_id"))
                .values("id", "cube_id", "user_id")
                .order_by("cube_id")
            )
        )
        user_id_in_db: set[int] = set([x["user_id"] for x in device_camera_list_in_db])

        # process
        if (user_id not in user_id_in_db) or (len(user_id_in_db) > 1):
            raise PermissionDenied
        Guest.objects.filter(id__in=request_data_camera_id_list).update(is_deleted=True)
        for cube_id_group in [
            list(g)
            for k, g in itertools.groupby(
                device_camera_list_in_db,
                lambda item: (item["cube_id"]),
            )
        ]:
            GlobalDeivceCamera.delete_camera_id_key_list_of_connect_status_dict(
                device_cube_id=cube_id_group[0]["cube_id"],
                device_camera_id_key_list=[x["id"] for x in cube_id_group],
            )

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=["get"], url_path="count")
    def count(self, request: Request, *args, **kwargs) -> Response:
        return Response(
            {"count": self.get_queryset().count()}, status=status.HTTP_200_OK
        )
