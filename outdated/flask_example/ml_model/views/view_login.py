"""
TODO: fix HTTP Method
"""
import string
import random
from urllib.request import Request
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.authtoken.models import Token

from ml_model.serializers.base import UserSerializer
from ml_model.serializers.device_cube import DeviceCubeSerializer
from ml_model.model.base import UserProfile
from django.contrib.auth.models import User

from ml_model.model.base import User
from ml_model.model.device_cube import DeviceCube

import logging

logger = logging.getLogger(__name__)


@api_view(["GET"])
def users_list(request):
    logger.debug("get request about users list")
    data = User.objects.all()
    serializers = UserSerializer(data, context={"request": request}, many=True)

    return Response(serializers.data)


@api_view(["POST"])
def get_id(request):
    logger.debug("get id using phone")

    data = request.data
    get_users_list: list[UserProfile] = (
        UserProfile.objects.select_related("user")
        .filter(phone_number=data["phone"])
        .all()
    )

    if len(get_users_list) == 0:
        return Response({"id": "", "err": "phone"})

    return Response({"id": get_users_list[0].user.username, "err": None})


@api_view(["POST"])
def get_pw(request: Request):
    logger.debug("get pw using id")
    data = request.data
    try:
        user_profile: UserProfile = (
            UserProfile.objects.select_related("user")
            .get(user__username=data["id"])
        )
    except ObjectDoesNotExist as e:
        return Response({"pw": "", "err": "id"})
    else:
        cube_list_info = DeviceCube.objects.filter(user_id=user_profile.user_id)

        cube_serializer = DeviceCubeSerializer(cube_list_info, many=True)
        cube_lists = []
        for cube in cube_serializer.data:
            cube_lists.append(cube['serial_number'])
        token = Token.objects.get(user_id=user_profile.user_id)
        return Response({"pw": user_profile.user.password, "err": None, "token": token.key, 'cube_lists':cube_lists})


@api_view(["POST"])
def register_user(request):
    logger.debug("register user")
    logger.debug(request.data)
    data = request.data

    id_list = UserProfile.objects.filter(user__username=data["id"])
    if 0 < len(id_list):
        return Response({"success": False, "why": "id"})

    phone_list = UserProfile.objects.filter(phone_number=data["phone"])
    if 0 < len(phone_list):
        return Response({"success": False, "why": "phone"})

    try:
        user = User(
            username=data["id"],
            email=data["id"],
            password=data["pw"]
        )
        user_profile = UserProfile(
            user=user,
            nickname="",
            phone_number=data["phone"],
            recent_camera_query=""
        )
        with transaction.atomic():
            user.save()
            user_profile.save()
        user.save()
    except Exception as ex:
        return Response({"success": False, "why": ex})
    else:
        token = Token.objects.create(user=user)
        return Response({"success": True, "why": None, "Token": token.key})


@api_view(["POST"])
def update_pw(request):
    logger.debug("update pw using phone")
    logger.debug(request.data)
    data = request.data
    user_profile: list[UserProfile] = UserProfile.objects.filter(phone_number=data["phone"])
    if len(user_profile) == 0:
        return Response({"success": False, "why": "phone"})
    try:
        User.objects.filter(id=user_profile[0].user_id).update(password=data["pw"])

    except Exception as ex:
        return Response({"success": False, "why": ex})
    else:
        return Response({"success": True, "why": None})


@api_view(["POST"])
def create_random_num(request):
    LENGTH = 6
    string_pool = string.digits
    result = ""

    for _ in range(LENGTH):
        result += random.choice(string_pool)

    logger.debug("create random number! {}".format(result))
    return Response(result)
