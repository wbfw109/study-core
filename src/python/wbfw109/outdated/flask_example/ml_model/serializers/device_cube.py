from typing import Optional, Union
from django.contrib.auth.hashers import make_password
from django.db.models.query import QuerySet
from django.http import Http404
from ml_model.model.base import User
from ml_model.model.device_cube import DeviceCube
from rest_framework import serializers, viewsets, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response


class DeviceCubeSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceCube
        fields = ["id", "name", "serial_number", "firmware"]
        read_only_fields = ["id", "serial_number", "firmware"]



class DeviceCubeRegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceCube
        fields = ["id", "name", "serial_number", "firmware"]
        read_only_fields = ["id", "name", "firmware"]


class DeviceCubeSetFirmwareSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceCube
        fields = ["id", "firmware"]
        read_only_fields = ["id"]



