from otree.api import *


class C(BaseConstants):
    NAME_IN_URL = 'consent'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    consent_1 = models.BooleanField(
        label='I confirm that I have read and understand the information sheet for the above study. '
              'I have had the opportunity to consider the information, ask questions and have had these answered satisfactorily.',
        widget=widgets.CheckboxInput,
        initial=False,
    )
    consent_2 = models.BooleanField(
        label='I understand that my participation is voluntary and that I am free to withdraw at any time '
              'without giving any reason and without penalty.',
        widget=widgets.CheckboxInput,
        initial=False,
    )
    consent_3 = models.BooleanField(
        label='I understand that data collected during the study may be looked at by individuals from '
              'The University of Warwick, from regulatory authorities, where it is relevant to my taking part '
              'in this study. I give permission for these individuals to have access to my data.',
        widget=widgets.CheckboxInput,
        initial=False,
    )
    consent_4 = models.BooleanField(
        label='I am happy for my data to be used in future research.',
        widget=widgets.CheckboxInput,
        initial=False,
    )
    consent_5 = models.BooleanField(
        label='I agree to take part in the above study.',
        widget=widgets.CheckboxInput,
        initial=False,
    )

    consent_given = models.StringField(blank=True, initial='no')

    prolific_ID = models.StringField(default=str(" "))


# PAGES

_CONSENT_ITEMS = [
    (1, 'I confirm that I have read and understand the information sheet for the above study. '
        'I have had the opportunity to consider the information, ask questions and have had these answered satisfactorily.'),
    (2, 'I understand that my participation is voluntary and that I am free to withdraw at any time '
        'without giving any reason and without penalty.'),
    (3, 'I understand that data collected during the study may be looked at by individuals from '
        'The University of Warwick, from regulatory authorities, where it is relevant to my taking part '
        'in this study. I give permission for these individuals to have access to my data.'),
    (4, 'I am happy for my data to be used in future research.'),
    (5, 'I agree to take part in the above study.'),
]


class Consent(Page):
    form_model = 'player'
    form_fields = ['consent_given']

    @staticmethod
    def is_displayed(player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player):
        return {'consent_items': _CONSENT_ITEMS}

    @staticmethod
    def before_next_page(player, timeout_happened):
        import random
        all_checked = (player.consent_given == 'yes')
        player.participant.vars['consent'] = all_checked

        if all_checked:
            # Assign condition with weights proportional to group size (oversample harder conditions)
            # A=1H+5LLM, B=2H+4LLM, C=3H+3LLM, D=4H+2LLM, E=5H+1LLM
            conditions = ['A', 'B', 'C', 'D', 'E']
            weights    = [  1,   2,   3,   4,   5]
            condition = random.choices(conditions, weights=weights, k=1)[0]
            condition_map = {'A': (1, 5), 'B': (2, 4), 'C': (3, 3), 'D': (4, 2), 'E': (5, 1)}
            num_humans, num_llms = condition_map[condition]
            player.participant.vars['condition']  = condition
            player.participant.vars['num_humans'] = num_humans
            player.participant.vars['num_llms']   = num_llms

        if player.participant.label:
            player.participant.vars['prolific_ID'] = player.participant.label


class Prolific(Page):
    form_model = 'player'
    form_fields = ['prolific_ID']

    @staticmethod
    def is_displayed(player):
        return (player.participant.vars.get('consent', False) == True and
                not player.participant.vars.get('prolific_ID'))

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.participant.vars['prolific_ID'] = player.prolific_ID


class PaymentInfo(Page):
    form_model = 'player'

    @staticmethod
    def is_displayed(player):
        return player.participant.vars.get('consent', False) == False

    @staticmethod
    def js_vars(player):
        return dict(
            completionLink=
            player.subsession.session.config.get('completionLink', '')
        )


page_sequence = [Consent, Prolific, PaymentInfo]
