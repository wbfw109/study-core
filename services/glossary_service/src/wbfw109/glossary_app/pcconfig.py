import pynecone as pc

config = pc.Config(
    app_name="glossary_app",
    db_url="sqlite:///pynecone.db",
)


# %shell> pc run --env dev
# %shell> pc run --env prod
# Title: production ~

# config = pc.Config(
#     app_name="glossary_app",
#     api_url="http://127.0.0.1:8000",
#     deploy_url="http://172.29.214.101:3001",
#     bun_path="$HOME/.bun/bin/bun",
#     db_url="sqlite:///pynecone.db",
# )
