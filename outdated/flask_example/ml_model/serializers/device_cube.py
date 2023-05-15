from ml_model.model.device_cube import DeviceCube
from rest_framework import serializers


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



