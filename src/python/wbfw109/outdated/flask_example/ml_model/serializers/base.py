from rest_framework import serializers
from ml_model.model.base import User
from ml_model.model.device_cube import DeviceCube
from django.contrib.auth.hashers import make_password


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User  # product 모델 사용
        # fields = '__all__'            # 모든 필드 포함
        fields = ("id", "name", "password", "phone_number", "created_datetime")
