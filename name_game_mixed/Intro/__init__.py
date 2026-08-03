from otree.api import *
import random


class C(BaseConstants):
    NAME_IN_URL = 'Intro'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


def _q4_choices(num_humans, num_llms):
    """Return 4 shuffled (value, label) tuples for Q4, with the correct answer included."""
    def make_label(h, l):
        h_str = f"{h} human participant" + ("s" if h > 1 else "")
        l_str = f"{l} AI" + ("s" if l > 1 else "")
        return f"{h_str} and {l_str}"

    def make_value(h, l):
        return f"{h}h_{l}ai"

    all_conditions = [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
    wrong = [(h, l) for h, l in all_conditions if h != num_humans]
    wrong_sample = random.sample(wrong, 3)

    choices = [(make_value(h, l), make_label(h, l)) for h, l in wrong_sample]
    choices.append((make_value(num_humans, num_llms), make_label(num_humans, num_llms)))
    random.shuffle(choices)
    return choices


class Player(BasePlayer):
    comprehension_check_attempts = models.IntegerField(initial=0)
    comprehension_check_failed = models.BooleanField(initial=False)

    comprehension_check_1 = models.StringField(
        label='What is your goal in each round?',
        choices=[
            ['choose_same',        'All 6 players in my group choose the same letter'],
            ['choose_different',   'Choose a different letter from other players'],
            ['choose_fastest',     'Choose your letter as quickly as possible'],
            ['choose_alphabetical','Choose letters in alphabetical order'],
        ],
        widget=widgets.RadioSelect
    )

    comprehension_check_2 = models.StringField(
        label='What happens to your group after each round?',
        choices=[
            ['stay_same',         'You stay in the same group of 6 players throughout all rounds'],
            ['choose_partner',    'You can choose your group members each round'],
            ['random_reshuffle',  'Groups are randomly reshuffled each round'],
            ['rotate_clockwise',  'Group members rotate in a fixed clockwise pattern'],
        ],
        widget=widgets.RadioSelect
    )

    comprehension_check_3 = models.StringField(
        label='How many rounds will you play in total?',
        choices=[
            ['5_rounds',  '5 rounds'],
            ['10_rounds', '10 rounds'],
            ['14_rounds', '14 rounds'],
            ['16_rounds', '16 rounds'],
        ],
        widget=widgets.RadioSelect
    )

    # Q4 is dynamic — rendered manually in the template; value submitted as plain POST field
    comprehension_check_4 = models.StringField(blank=True)


# PAGES

class Intro(Page):
    form_model = 'player'
    form_fields = ['comprehension_check_1', 'comprehension_check_2',
                   'comprehension_check_3', 'comprehension_check_4']

    @staticmethod
    def is_displayed(player):
        return (player.round_number == 1 and
                player.participant.vars.get('consent', False) == True and
                not player.comprehension_check_failed and
                player.comprehension_check_attempts == 0)

    @staticmethod
    def vars_for_template(player):
        num_humans = player.participant.vars.get('num_humans', 5)
        num_llms   = player.participant.vars.get('num_llms', 1)
        return {
            'num_humans':    num_humans,
            'num_llms':      num_llms,
            'total_players': num_humans + num_llms,
            'q4_choices':    _q4_choices(num_humans, num_llms),
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.comprehension_check_attempts += 1
        num_humans = player.participant.vars.get('num_humans', 5)
        num_llms   = player.participant.vars.get('num_llms', 1)
        correct_q4 = f"{num_humans}h_{num_llms}ai"

        wrong_questions = []
        if player.comprehension_check_1 != 'choose_same':
            wrong_questions.append('1')
        if player.comprehension_check_2 != 'stay_same':
            wrong_questions.append('2')
        if player.comprehension_check_3 != '14_rounds':
            wrong_questions.append('3')
        if player.comprehension_check_4 != correct_q4:
            wrong_questions.append('4')

        player.participant.vars['first_attempt_passed'] = len(wrong_questions) == 0
        player.participant.vars['wrong_questions'] = wrong_questions


class Intro_2(Page):
    form_model = 'player'
    form_fields = ['comprehension_check_1', 'comprehension_check_2',
                   'comprehension_check_3', 'comprehension_check_4']

    @staticmethod
    def is_displayed(player):
        return (player.round_number == 1 and
                player.participant.vars.get('consent', False) == True and
                not player.comprehension_check_failed and
                player.comprehension_check_attempts == 1 and
                not player.participant.vars.get('first_attempt_passed', True))

    @staticmethod
    def vars_for_template(player):
        num_humans = player.participant.vars.get('num_humans', 5)
        num_llms   = player.participant.vars.get('num_llms', 1)
        return {
            'wrong_questions': player.participant.vars.get('wrong_questions', []),
            'num_humans':      num_humans,
            'num_llms':        num_llms,
            'total_players':   num_humans + num_llms,
            'q4_choices':      _q4_choices(num_humans, num_llms),
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.comprehension_check_attempts += 1
        num_humans = player.participant.vars.get('num_humans', 5)
        num_llms   = player.participant.vars.get('num_llms', 1)
        correct_q4 = f"{num_humans}h_{num_llms}ai"

        answers_correct = (
            player.comprehension_check_1 == 'choose_same' and
            player.comprehension_check_2 == 'stay_same' and
            player.comprehension_check_3 == '14_rounds' and
            player.comprehension_check_4 == correct_q4
        )
        if not answers_correct:
            player.comprehension_check_failed = True


class Start(Page):
    @staticmethod
    def is_displayed(player):
        return (player.participant.vars.get('consent', False) == True and
                not player.comprehension_check_failed)

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.participant.vars['attention_check_failed'] = player.comprehension_check_failed


class PaymentInfo(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return (player.participant.vars.get('consent', False) == False or
                player.comprehension_check_failed)

    @staticmethod
    def js_vars(player):
        return dict(
            completionLink=player.subsession.session.config['completionLink']
        )


page_sequence = [Intro, Intro_2, Start, PaymentInfo]
