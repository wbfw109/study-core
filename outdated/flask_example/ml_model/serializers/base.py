from rest_framework import serializers
from ml_model.model.base import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User  # product 모델 사용
        # fields = '__all__'            # 모든 필드 포함
        fields = ("id", "name", "password", "phone_number", "created_datetime")
