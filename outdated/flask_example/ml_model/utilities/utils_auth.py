from typing import Union
from django.db.models.query import QuerySet
from ml_model.utilities.utils_typing import get_dict_from_attribute_to_object
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

def create_superuser_tokens() -> None:
    superuser_list: list[Union[QuerySet, User]] = User.objects.filter(is_superuser=True).all()
    user_dict_from_id_to_object = get_dict_from_attribute_to_object(objects=superuser_list, attribute_name="id")

    superuser_id_list: list[int] = [superuser.id for superuser in superuser_list]
    superuser_id_list_in_token: list[Union[QuerySet, User]]= Token.objects.filter(user_id__in=superuser_id_list).values_list("user_id", flat=True)
    for superuser_id_to_be_created in set(superuser_id_list) - set(superuser_id_list_in_token):
        Token.objects.create(user=user_dict_from_id_to_object[superuser_id_to_be_created])
