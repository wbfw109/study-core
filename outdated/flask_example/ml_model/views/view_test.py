from django.db import transaction
from django.db.utils import DatabaseError
from rest_framework.authtoken.models import Token
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from ml_model.model.base import User, UserProfile
from django.contrib.auth.hashers import make_password
import logging


class SignupView(APIView):
    def post(self, request: Request):
        try:
            user = User(
                username=request.data["id"],
                email=request.data["email"],
                password=make_password(request.data["password"]),
            )
            user_profile = UserProfile(
                user=user,
                nickname=request.data["nickname"],
                phone_number=request.data["phone_number"],
                recent_camera_query=""
            )
            with transaction.atomic():
                user.save()
                user_profile.save()
        except DatabaseError as e:
            logging.error(e)
            return Response()
        else:
            token = Token.objects.create(user=user)
            return Response({"Token": token.key})
