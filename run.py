# .envを読み込む
from dotenv import load_dotenv
load_dotenv()

from app import create_app, db
from flask_migrate import Migrate

app = create_app()
migrate = Migrate(app, db)

from flask.cli import with_appcontext
import click

@click.command(name='create_tables')
@with_appcontext
def create_tables():
    db.create_all()

# CLIコマンドとして登録（なくてもよいがCLIの登録が有効になるテスト用）
app.cli.add_command(create_tables)
