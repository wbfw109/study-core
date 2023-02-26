# %%
#
import dataclasses
from ipaddress import ip_address
import IPython
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
import logging
import os
from django.db.models.expressions import F

from django.db.models.query import QuerySet

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

from typing import (
    Any,
    ClassVar,
    Iterable,
    NamedTuple,
    Tuple,
    Type,
    TypedDict,
    Union,
    Optional,
)
from pathlib import Path
import pprint

pp = pprint.PrettyPrinter(compact=False)

from backend import development_settings
import collections, itertools
import copy
import datetime, time
import django
import inspect
import math, random
import os
import re
import shutil
import xml.etree.ElementTree as ET

os.environ["DJANGO_SETTINGS_MODULE"] = "backend.development_settings"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()
from ml_model.global_object import GlobalDeivceCamera, GlobalConfig
from ml_model.model.base import ConnectStatus, ThreatEvent, UserProfile
from ml_model.model.device_camera import (
    DeviceCamera,
    DeviceCameraEvent,
    DeviceCameraConnectStatus,
)
import string

from ml_model.model.device_cube import DeviceCube, Guest, DeviceCubeAlertsGuest
from ml_model.serializers.device_camera import DeviceCameraSerializer
from rest_framework import views, viewsets
from django.contrib.auth.hashers import make_password
from django.db import DatabaseError, transaction
from rest_framework.authtoken.models import Token
import itertools
from django.contrib.auth.models import User
from django.db.models.expressions import F
from django.db.models.query import QuerySet
from django.db.models.query_utils import Q
from django.http import Http404
from django.core.exceptions import PermissionDenied
from io import BytesIO
from ml_model.global_object import (
    GlobalConfig,
    GlobalDeivceCamera,
    GlobalResponseMessage,
)
from ml_model.model.device_camera import DeviceCamera, DeviceCameraEvent
from ml_model.model.device_cube import DeviceCube, DeviceCubeAlertsGuest, Guest
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

GlobalDeivceCamera.initialize_dict()

#%%
class ThreatEventSerializer(serializers.ModelSerializer):
    """It is used to CRUD in page of threat_event."""

    class Meta:
        model = ThreatEvent
        fields = ["id", "name"]

    def create(self, validated_data: dict):
        print(validated_data)
        print(type(validated_data))
        return


data = [{"name": "normal"}, {"name": "fire"}]
serializer = ThreatEventSerializer(data=data, many=True)

serializer.is_valid()
serializer.save()


#%%
DeviceCamera.objects.all().count()
#%%

user_profile: Union[QuerySet, UserProfile] = UserProfile.objects.select_related(
    "user"
).get(id=5)
user_profile2: Union[QuerySet, UserProfile] = UserProfile.objects.select_related(
    "user"
).get(id=6)
user_profile2.nickname = "wbfw109_3333_nick"
print("=============")
UserProfile.objects.bulk_update(objs=[user_profile, user_profile2], fields=["nickname"])


#%%
from typing import TypedDict


class ThisData(TypedDict):
    id: int
    ip_address: str
    device_id: str
    device_pw: str
    location: str
    cube_id: int


x = ThisData(
    ip_address="hhihi", device_pw="device_pppw", location="looocation", cube_id=123
)
validate_iterable_comply_optional_typed_dict([x], ThisData)

# .filter(id__in=[1, 2, 3])
# user = User.objects.filter(id=9)

# Guest.objects.filter(user_id= user.id)

#%%

## test data
# user
try:
    user: User = User(
        username="wbfw109_1",
        email="wbfw109_1.park@triplllet.com",
        password=make_password("wbfw1091234"),
    )
    user_profile = UserProfile(
        user=user,
        nickname="wbfw109_1_nick",
        phone_number="010-1234-5678",
        recent_camera_query="",
    )
    with transaction.atomic():
        user.save()
        user_profile.save()
except DatabaseError as e:
    print(e)
else:
    Token.objects.create(user=user)
#%%
# device_cube
count_to_be_created = 10


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

    if len(serial_list) == count_to_be_created:
        break
serial_list = list(serial_list)
device_cube_list: list[DeviceCube] = DeviceCube.objects.bulk_create(
    [
        DeviceCube(
            user_id=user, serial_number=serial_list[i], firmware="1.0", name="rrrr"
        )
        for i in range(count_to_be_created)
    ]
)
device_cube_of_user_count: int = 3
some_device_cube_list: list[DeviceCube] = []
for i, device_cube in enumerate(device_cube_list, start=1):
    device_cube.user = user
    device_cube.save()
    some_device_cube_list.append(device_cube)
    if i >= device_cube_of_user_count:
        break

# device_camera
device_camera_list: list[DeviceCube] = DeviceCamera.objects.bulk_create(
    [
        DeviceCamera(
            cube=device_cube,
            ip_address="10.1234.1234",
            device_id="device_id1111",
            device_pw="12device_pw",
            location="locaiotnnnn",
        )
        for device_cube in some_device_cube_list
    ]
)
#%%
user = User.objects.get(id=9)

guest_list = Guest.objects.bulk_create(
    [Guest(user=user, phone_number=f"010-1234-567{i}") for i in range(10)]
)
#%%
"""
image field.. 정상적으로 들어오고 경로 어떻게 저장되는지 확인,

from dummy
# pass serializers to frontend
# alert for device_camera_real_threat_event_list

mqtt
threat event table (id, name) 을 보내주는 APIView 필요할듯.
serializer update (multiple), delete (multiple) 해야 함.
camera 변경에 따른 global key 관리................................ 
    생성 -> create
    삭제 -> delete



serializer naming

"""
