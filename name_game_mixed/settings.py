from os import environ
from dotenv import load_dotenv
load_dotenv()

SESSION_CONFIGS = [
    dict(
        name='name_game_mixed',
        display_name='Name Game Mixed (Human + AI)',
        app_sequence=['consent', 'Intro', 'name_game', 'End_name'],
        num_demo_participants=5,
        use_browser_bots=False,
        use_live=True,
        test_mode=False,
        completionLink='https://app.prolific.com/submissions/complete?cc=C1QS7X7J',
    )
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

ROOMS = [
    dict(
        name='name_game_mixed',
        display_name='Name Game Mixed',
        # participant_label_file='_rooms/study.txt',
        # use_secure_urls=True,
    ),
]

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

LANGUAGE_CODE = 'en'

REAL_WORLD_CURRENCY_CODE = 'GBP'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '3847291058374'
