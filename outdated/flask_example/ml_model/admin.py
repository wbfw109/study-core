# Register your models here.
from django.contrib import admin
from django.db import models
from ml_model.model.device_camera import DeviceCamera
from ml_model.model.device_cube import DeviceCube
from ml_model.model.base import UserProfile
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'profiles'

class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)


class DeviceCameraInline(admin.StackedInline):
    model = DeviceCamera
    extra = 3
    classes = ("collapse",)


class DeviceCubeAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {"fields": ["user_id", "serial_number", "firmware", "name"]}),
    ]
    inlines = [DeviceCameraInline]
    list_display = (
        "id",
        "user_id",
        "serial_number",
        "firmware",
        "name",
        "created_datetime",
    )
    list_filter = ["created_datetime"]
    search_fields = ["serial_number"]

admin.site.register(DeviceCube, DeviceCubeAdmin)
