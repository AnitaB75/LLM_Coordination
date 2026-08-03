from otree.api import *
import json


class C(BaseConstants):
    NAME_IN_URL = 'End_name'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Q1: AI tools — stored as JSON list (e.g. '["ChatGPT (OpenAI)", "Other: Copilot"]')
    ai_tools = models.LongStringField(blank=True)

    # Q2: Frequency per tool — stored as JSON dict e.g. {"ChatGPT (OpenAI)": "Daily", ...}
    ai_frequencies_json = models.LongStringField(blank=True)

    # Q3: Gender
    gender = models.StringField(
        choices=['Male', 'Female', 'Non-binary', 'Other', 'Prefer not to say'],
        widget=widgets.RadioSelect,
        blank=True,
        label='What is your gender?',
    )

    # Q4: Age (optional)
    age = models.IntegerField(
        blank=True,
        min=18,
        max=100,
        label='What is your age? (Enter a number, or leave blank if you prefer not to say)',
    )

    # Moral appropriateness — self (0–100 sliders, submitted via hidden inputs in template)
    moral_always_j = models.IntegerField(min=0, max=100, blank=True)
    moral_j_then_m = models.IntegerField(min=0, max=100, blank=True)
    moral_j_then_n = models.IntegerField(min=0, max=100, blank=True)

    # Moral appropriateness — teammate (0–100 sliders)
    teammate_moral_always_j = models.IntegerField(min=0, max=100, blank=True)
    teammate_moral_j_then_m = models.IntegerField(min=0, max=100, blank=True)
    teammate_moral_j_then_n = models.IntegerField(min=0, max=100, blank=True)



# ── helpers ──────────────────────────────────────────────────────────────────

def _standard_displayed(player):
    """Standard condition: shown to consenters (including timed-out) and failed-attention."""
    return (
        (player.participant.vars.get('consent', False) == True and
         not player.participant.vars.get('attention_check_failed', False)) or
        player.participant.vars.get('timed_out', False) or
        player.participant.vars.get('matching_timeout', False) or
        player.participant.vars.get('attention_check_failed', False)
    )


def _selected_tools(player):
    """Return list of selected tool names (empty if None or blank)."""
    raw = player.ai_tools
    if not raw:
        return []
    try:
        tools = json.loads(raw)
        return [t for t in tools if t != 'None']
    except Exception:
        return []


# ── PAGES ─────────────────────────────────────────────────────────────────────

class End_Section(Page):
    @staticmethod
    def is_displayed(player):
        return _standard_displayed(player)

    @staticmethod
    def vars_for_template(player):
        total_points = player.participant.vars.get('total_points', 0)
        return {
            'total_points': total_points,
            'total_pay': f"£{total_points / 100:.2f}",
        }


class AIToolsPage(Page):
    form_model = 'player'
    form_fields = ['ai_tools']

    @staticmethod
    def is_displayed(player):
        return (
            player.participant.vars.get('consent', False) == True and
            not player.participant.vars.get('attention_check_failed', False)
        ) or player.participant.vars.get('timed_out', False)


class AIFrequencyPage(Page):
    form_model = 'player'
    form_fields = ['ai_frequencies_json']

    @staticmethod
    def is_displayed(player):
        if not (
            (player.participant.vars.get('consent', False) == True and
             not player.participant.vars.get('attention_check_failed', False)) or
            player.participant.vars.get('timed_out', False)
        ):
            return False
        return len(_selected_tools(player)) > 0

    @staticmethod
    def vars_for_template(player):
        return {
            'tools': _selected_tools(player),
            'frequencies': ['Daily', 'Weekly', 'Monthly', 'Rarely'],
        }


def _post_game_displayed(player):
    """Shown to consented non-matching-timeout participants (and timed-out players)."""
    return (
        (player.participant.vars.get('consent', False) == True and
         not player.participant.vars.get('attention_check_failed', False) and
         not player.participant.vars.get('matching_timeout', False)) or
        player.participant.vars.get('timed_out', False)
    )


class MoralAppropriatenessQuestions(Page):
    form_model = 'player'
    form_fields = ['moral_always_j', 'moral_j_then_m', 'moral_j_then_n']

    @staticmethod
    def is_displayed(player):
        return _post_game_displayed(player)


class TeammateMoralAppropriatenessQuestions(Page):
    form_model = 'player'
    form_fields = ['teammate_moral_always_j', 'teammate_moral_j_then_m', 'teammate_moral_j_then_n']

    @staticmethod
    def is_displayed(player):
        return _post_game_displayed(player)


class Demographics(Page):
    form_model = 'player'
    form_fields = ['gender', 'age']

    @staticmethod
    def is_displayed(player):
        return _standard_displayed(player)


class Debrief(Page):
    @staticmethod
    def is_displayed(player):
        return (
            player.participant.vars.get('consent', False) == True and
            not player.participant.vars.get('attention_check_failed', False)
        ) or player.participant.vars.get('timed_out', False)

    @staticmethod
    def vars_for_template(player):
        total_points = player.participant.vars.get('total_points', 0)
        return {
            'total_points': total_points,
            'total_pay': f"£{total_points / 100:.2f}",
        }


class PaymentInfo(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return (
            player.participant.vars.get('consent', False) == True or
            player.participant.vars.get('timed_out', False) or
            player.participant.vars.get('matching_timeout', False) or
            player.participant.vars.get('attention_check_failed', False)
        )

    @staticmethod
    def js_vars(player):
        return dict(completionLink=player.subsession.session.config['completionLink'])


page_sequence = [
    End_Section,
    AIToolsPage,
    AIFrequencyPage,
    MoralAppropriatenessQuestions,
    TeammateMoralAppropriatenessQuestions,
    Demographics,
    Debrief,
    PaymentInfo,
]
