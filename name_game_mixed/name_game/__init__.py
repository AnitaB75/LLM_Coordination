from otree.api import *
import os
import json
import asyncio
import random
from pydantic import BaseModel
from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError, RateLimitError


# ── OpenAI ─────────────────────────────────────────────────────────────────────

MODEL_NAME = 'gpt-5.5'
REASONING_LVL = 'medium'
VALID_LETTERS = {'J', 'P', 'Q', 'R', 'N', 'M', 'F', 'C', 'X', 'Y'}

CONDITION_HUMANS = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
CONDITION_LLMS   = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        print(f"[LLM] Initializing client. API key present: {bool(api_key)}, prefix: {api_key[:8] if api_key else 'MISSING'}")
        _client = AsyncOpenAI(api_key=api_key)
        print(f"[LLM] Client initialized.")
    return _client


class LLMLetterOutput(BaseModel):
    letter: str


async def _api_call(messages, round_number=None):
    """Call OpenAI Responses API; returns (parsed_result, reasoning_summary)."""
    import time as _time
    t0 = _time.time()
    round_tag = f"R{round_number} " if round_number else ""
    for attempt in range(3):
        try:
            if attempt == 0:
                print(f"[LLM] {round_tag}sending request...")
            else:
                print(f"[LLM] {round_tag}Attempt {attempt + 1}: sending request...")
            kwargs = dict(
                model=MODEL_NAME,
                input=messages,
                text_format=LLMLetterOutput,
                max_output_tokens=1500,
            )
            if REASONING_LVL != 'none':
                kwargs['reasoning'] = {'effort': REASONING_LVL}
            response = await _get_client().responses.parse(**kwargs)
            parsed = response.output_parsed
            if parsed is None:
                print(f"[LLM] {round_tag}Attempt {attempt + 1}: output_parsed is None — retrying...")
                if attempt == 2:
                    raise ValueError("output_parsed returned None after 3 attempts")
                await asyncio.sleep(2 ** attempt)
                continue
            elapsed = _time.time() - t0
            print(f"[LLM] {round_tag}Success. Letter={parsed.letter} | "
                  f"tokens: {response.usage.input_tokens}in / {response.usage.output_tokens}out | "
                  f"elapsed: {elapsed:.1f}s")
            return parsed, ''
        except RateLimitError as e:
            print(f"[LLM] Attempt {attempt + 1} RATE LIMITED: {e}")
            if attempt == 2:
                raise
            retry_after = None
            if hasattr(e, 'response') and e.response is not None:
                retry_after = e.response.headers.get('retry-after')
            wait = float(retry_after) if retry_after else 10 * (2 ** attempt)
            print(f"[LLM] Backing off {wait}s before retry...")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"[LLM] Attempt {attempt + 1} FAILED: {type(e).__name__}: {e}")
            if attempt == 2 or isinstance(e, (AuthenticationError, PermissionDeniedError)):
                raise
            await asyncio.sleep(2 ** attempt)


def _build_system_prompt(num_humans, num_llms):
    total = num_humans + num_llms
    other_ais = num_llms - 1

    if num_humans == 1:
        human_desc = "1 human"
    else:
        human_desc = f"{num_humans} humans"

    if other_ais == 0:
        ai_desc = "0 other AIs"
    elif other_ais == 1:
        ai_desc = "1 other AI"
    else:
        ai_desc = f"{other_ais} other AIs"

    return f"""Coordination game: {total} players ({human_desc}, {ai_desc}). Each round everyone simultaneously picks a letter. Points only if ALL choose the same letter. 14 rounds, same group, no communication.

Letters: J P Q R N M F C X Y
History format: You=your choice, H=[human choices], A=[other AI choices], Σ=your cumulative points."""


def _fetch_round_history(group, num_humans, num_llms):
    """Fetch all past-round data once, returning a list of per-round dicts.
    Called once per group per round; results are shared across all LLM instances."""
    total = num_humans + num_llms
    rounds = []
    for r in range(1, group.round_number):
        past_group = group.in_round(r)
        human_choices = [p.letter_choice for p in past_group.get_players() if p.letter_choice]
        raw = past_group.llm_choices_json
        llm_choices = json.loads(raw) if (raw and raw not in ('', 'PENDING')) else ['?'] * num_llms
        all_choices = human_choices + llm_choices
        all_same = (
            len(set(all_choices)) == 1
            and len(all_choices) == total
            and '?' not in all_choices
        )
        if all_same:
            letter = all_choices[0]
            pts = 10 if letter == 'J' else (15 if letter in ('M', 'N') and r > 7 else 0)
        else:
            pts = 0
        rounds.append({'r': r, 'human_choices': human_choices, 'llm_choices': llm_choices,
                       'all_same': all_same, 'pts': pts})
    return rounds


def _build_llm_messages(round_history, llm_index, num_humans, num_llms, round_number):
    """Build message list for one LLM instance from pre-fetched history (no DB calls)."""
    history_lines = []
    llm_cumulative = 0
    for rd in round_history:
        my_choice = rd['llm_choices'][llm_index] if llm_index < len(rd['llm_choices']) else '?'
        other_llm_choices = [c for i, c in enumerate(rd['llm_choices']) if i != llm_index]
        llm_cumulative += rd['pts']
        h_str = "[" + ",".join(rd['human_choices']) + "]" if rd['human_choices'] else "[]"
        a_str = "[" + ",".join(other_llm_choices) + "]" if other_llm_choices else "[]"
        match_str = f"MATCH +{rd['pts']}" if rd['all_same'] else "no match"
        history_lines.append(
            f"R{rd['r']}: You={my_choice} H={h_str} A={a_str} → {match_str} (Σ{llm_cumulative})"
        )

    history_text = (
        "PREVIOUS ROUNDS:\n" + "\n".join(history_lines)
        if history_lines
        else "This is the first round — no history yet."
    )
    if round_number > 7:
        payoff_line = "This round: J=10pts, M=15pts, N=15pts if all match. All other letters=0pts."
    else:
        payoff_line = "This round: J=10pts if all match. All other letters=0pts."
    user_message = (
        f"It is now Round {round_number} of 14.\n\n"
        f"{payoff_line}\n\n"
        f"{history_text}\n\n"
        f"Please choose your letter for Round {round_number}."
    )

    return [
        {"role": "system", "content": _build_system_prompt(num_humans, num_llms)},
        {"role": "user", "content": user_message},
    ]


# ── oTree ──────────────────────────────────────────────────────────────────────

class C(BaseConstants):
    NAME_IN_URL = 'name_game_mixed'
    PLAYERS_PER_GROUP = 5  # maximum humans per group; actual size set per condition
    NUM_ROUNDS = 14

    letter_choices = ['Q', 'M', 'N', 'X', 'Y', 'F', 'J', 'P', 'R', 'C']
    letters_early = ['J']
    letters_late = ['J', 'M', 'N']


class Subsession(BaseSubsession):
    def group_by_arrival_time_method(self, waiting_players):
        """Group N human players sharing the same condition; LLMs fill the rest."""
        import time
        from collections import defaultdict

        current_time = time.time()
        for p in waiting_players:
            if 'wait_page_arrival' not in p.participant.vars:
                p.participant.vars['wait_page_arrival'] = current_time

        # Try to form a complete group for any condition
        by_condition = defaultdict(list)
        for p in waiting_players:
            cond = p.participant.vars.get('condition', 'E')
            by_condition[cond].append(p)

        for cond, players in by_condition.items():
            needed = CONDITION_HUMANS[cond]
            if len(players) >= needed:
                return players[:needed]

        # Release any player who has been waiting over 10 minutes
        if waiting_players:
            all_timed_out = all(
                (current_time - p.participant.vars.get('wait_page_arrival', current_time)) > 600
                for p in waiting_players
            )
            if all_timed_out:
                return waiting_players

        return None


class Group(BaseGroup):
    # '' = not started, 'PENDING' = in flight, JSON list = done
    llm_choices_json = models.LongStringField(default='')
    llm_fallbacks_json = models.LongStringField(default='')
    num_humans = models.IntegerField(default=5)
    num_llms = models.IntegerField(default=1)
    submissions = models.IntegerField(default=0)
    someone_timed_out = models.BooleanField(default=False)
    llm_fallback = models.BooleanField(default=False)


class Player(BasePlayer):
    letter_choice = models.StringField(default='')
    round_points = models.IntegerField(default=0)
    total_points = models.IntegerField(default=0)

    def calculate_points(self):
        """All human choices + all LLM choices must match."""
        num_humans = self.participant.vars.get('num_humans', 5)
        num_llms = self.participant.vars.get('num_llms', 1)

        human_players = [p for p in self.group.get_players()
                         if not p.participant.vars.get('timed_out', False)]
        human_choices = [p.letter_choice for p in human_players if p.letter_choice]

        raw = self.group.llm_choices_json
        if not raw or raw == 'PENDING':
            return 0
        llm_choices = json.loads(raw)

        if len(human_choices) != num_humans or len(llm_choices) != num_llms:
            return 0

        all_choices = human_choices + llm_choices
        if len(set(all_choices)) != 1:
            return 0

        letter = self.letter_choice
        if self.round_number <= 7:
            return 10 if letter == 'J' else 0
        else:
            return 10 if letter == 'J' else (15 if letter in ('M', 'N') else 0)

    def get_available_point_values(self):
        if self.round_number <= 7:
            return {'J': 10, 'P': 0, 'R': 0, 'C': 0, 'Q': 0, 'M': 0, 'N': 0, 'X': 0, 'Y': 0, 'F': 0}
        else:
            return {'J': 10, 'M': 15, 'N': 15, 'P': 0, 'R': 0, 'C': 0, 'Q': 0, 'X': 0, 'Y': 0, 'F': 0}


# HELPER FUNCTIONS

def _check_and_advance(group):
    """Broadcast advance when LLMs have all responded AND all humans have submitted."""
    llm_done = group.llm_choices_json and group.llm_choices_json != 'PENDING'
    humans_done = group.submissions >= group.num_humans
    print(f"[Check] Round {group.round_number}: llm_done={llm_done}, "
          f"humans_done={humans_done} ({group.submissions}/{group.num_humans})")
    if llm_done and humans_done:
        players = group.get_players()
        print(f"[Check] Sending advance to player ids: {[p.id_in_group for p in players]}")
        return {p.id_in_group: {'type': 'advance'} for p in players}
    return None


def handle_player_timeout(player):
    player.participant.vars['timed_out'] = True
    player.participant.vars['experiment_ended'] = True
    player.participant.vars['is_active'] = False


def handle_matching_timeout(player):
    player.participant.vars['matching_timeout'] = True
    player.participant.vars['experiment_ended'] = True


def get_active_players(group):
    return [p for p in group.get_players() if not p.participant.vars.get('timed_out', False)]


# PAGES

class WaitForPlayers(WaitPage):
    group_by_arrival_time = True
    timeout_seconds = 600

    @staticmethod
    def is_displayed(player):
        test_mode = player.session.config.get('test_mode', False)
        if test_mode:
            return False
        return (player.round_number == 1 and
                not player.participant.vars.get('timed_out', False) and
                player.participant.vars.get('consent', False) == True and
                not player.participant.vars.get('attention_check_failed', False))

    @staticmethod
    def before_next_page(player, timeout_happened):
        if timeout_happened:
            handle_matching_timeout(player)

    @staticmethod
    def after_all_players_arrive(group):
        players = group.get_players()
        group_size = len(players)
        cond = players[0].participant.vars.get('condition', 'E')
        required = CONDITION_HUMANS.get(cond, 5)

        if group_size < required:
            for p in players:
                p.participant.vars['matching_timeout'] = True
                p.participant.vars['experiment_ended'] = True
        else:
            num_humans = group_size
            num_llms = CONDITION_LLMS.get(cond, 1)
            group.num_humans = num_humans
            group.num_llms = num_llms
            for p in players:
                p.participant.vars['is_active'] = True
                p.participant.vars['timed_out'] = False
                # Cache on participant so rounds 2-14 can read it
                p.participant.vars['num_humans'] = num_humans
                p.participant.vars['num_llms'] = num_llms


class Name_Game(Page):
    form_model = 'player'
    form_fields = []
    timeout_seconds = 120

    @staticmethod
    def before_next_page(player, timeout_happened):
        group = player.group
        all_players = group.get_players()

        if timeout_happened and not player.letter_choice:
            group.someone_timed_out = True
        # Any human with no letter_choice means the group ended early
        if any(not p.letter_choice for p in all_players):
            group.someone_timed_out = True
        # LLM failure also ends the group
        if group.llm_fallback:
            group.someone_timed_out = True
        # Propagate from prior rounds
        if any(p.participant.vars.get('timed_out', False) for p in all_players):
            group.someone_timed_out = True

        if group.someone_timed_out:
            handle_player_timeout(player)

    def is_displayed(player):
        return (not player.participant.vars.get('timed_out', False) and
                not player.participant.vars.get('matching_timeout', False) and
                player.participant.vars.get('consent', False) == True and
                not player.participant.vars.get('attention_check_failed', False))

    def vars_for_template(player):
        current_total = player.participant.vars.get('total_points', 0)
        num_humans = player.participant.vars.get('num_humans', 5)
        num_llms = player.participant.vars.get('num_llms', 1)
        total_players = num_humans + num_llms

        return {
            'round_num': player.round_number,
            'current_total_points': current_total,
            'total_players': total_players,
            'num_humans': num_humans,
            'num_llms': num_llms,
        }

    @staticmethod
    async def live_method(player, data):
        group = player.group
        msg_type = data.get('type')

        # Sync group-level condition fields from participant (they reset each round)
        num_humans = player.participant.vars.get('num_humans', 5)
        num_llms = player.participant.vars.get('num_llms', 1)
        group.num_humans = num_humans
        group.num_llms = num_llms

        if msg_type == 'page_loaded':
            print(f"[Game] Round {group.round_number}: Player {player.id_in_group} page loaded. "
                  f"llm_choices_json='{group.llm_choices_json[:40] if group.llm_choices_json else ''}'")
            all_players = group.get_players()

            # If any group member was timed out in a prior round, end this round immediately
            if group.someone_timed_out or any(p.participant.vars.get('timed_out', False) for p in all_players):
                print(f"[Timeout] Round {group.round_number}: prior timeout detected on page_load — advancing all.")
                group.someone_timed_out = True
                yield {p.id_in_group: {'type': 'advance'} for p in all_players}
                return

            # If PENDING, a call is already in flight — wait for it to complete.
            # If the server restarted and killed the task, the 120s page timeout
            # will fire before_next_page(timeout_happened=True) and redirect the group.
            if group.llm_choices_json == 'PENDING':
                print(f"[LLM] Round {group.round_number}: call already in flight — not re-firing.")
                return

            if group.llm_choices_json != '':
                advance = _check_and_advance(group)
                if advance:
                    yield advance
                return

            # First page_load: fire all LLM calls in parallel
            group.llm_choices_json = 'PENDING'
            # Jitter desynchronises groups that advance in lockstep, spreading
            # their API calls across the minute rather than spiking all at once
            jitter = random.uniform(0, 3)
            print(f"[LLM] Round {group.round_number}: jitter {jitter:.2f}s before firing...")
            await asyncio.sleep(jitter)
            print(f"[LLM] Round {group.round_number}: starting {num_llms} parallel API call(s)...")

            round_history = _fetch_round_history(group, num_humans, num_llms)
            messages_list = [
                _build_llm_messages(round_history, i, num_humans, num_llms, group.round_number)
                for i in range(num_llms)
            ]

            try:
                results = await asyncio.gather(
                    *[asyncio.wait_for(_api_call(msgs, round_number=group.round_number), timeout=120)
                      for msgs in messages_list],
                    return_exceptions=True
                )

                choices = []
                fallbacks = []
                any_fallback = False

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"[LLM] Instance {i} FAILED: {type(result).__name__}: {result}. Falling back to J.")
                        choices.append('J')
                        fallbacks.append(True)
                        any_fallback = True
                    else:
                        parsed, _ = result
                        letter = parsed.letter.strip().upper()
                        if letter not in VALID_LETTERS:
                            print(f"[LLM] Instance {i} returned invalid letter '{letter}', falling back to J.")
                            choices.append('J')
                            fallbacks.append(True)
                            any_fallback = True
                        else:
                            choices.append(letter)
                            fallbacks.append(False)
                            print(f"[LLM] Instance {i} chose '{letter}'")

                group.llm_choices_json = json.dumps(choices)
                group.llm_fallbacks_json = json.dumps(fallbacks)
                group.llm_fallback = any_fallback

                if any_fallback:
                    print(f"[LLM] Round {group.round_number}: fallback occurred — redirecting group to end.")
                    group.someone_timed_out = True
                    yield {p.id_in_group: {'type': 'advance'} for p in all_players}
                    return

            except Exception as e:
                print(f"[LLM] Unexpected error: {type(e).__name__}: {e}. Redirecting group to end.")
                group.llm_choices_json = json.dumps(['?'] * num_llms)
                group.llm_fallbacks_json = json.dumps([True] * num_llms)
                group.llm_fallback = True
                group.someone_timed_out = True
                yield {p.id_in_group: {'type': 'advance'} for p in all_players}
                return

            # If a human timed out while the LLM was in flight, redirect everyone
            if group.someone_timed_out:
                print(f"[LLM] Round {group.round_number}: human timed out while LLM was in flight — redirecting all.")
                yield {p.id_in_group: {'type': 'advance'} for p in all_players}
                return

            advance = _check_and_advance(group)
            if advance:
                yield advance

        elif msg_type == 'player_timeout':
            if not player.letter_choice:
                print(f"[Timeout] Round {group.round_number}: Player {player.id_in_group} timed out.")
                group.someone_timed_out = True
                yield {p.id_in_group: {'type': 'advance'} for p in group.get_players()}

        elif msg_type == 'ping':
            if group.someone_timed_out:
                print(f"[Timeout] Round {group.round_number}: Player {player.id_in_group} notified via ping.")
                yield {player.id_in_group: {'type': 'advance'}}

        elif msg_type == 'letter_chosen':
            letter = data.get('letter', '').strip().upper()
            if letter not in VALID_LETTERS:
                yield {player.id_in_group: {'type': 'error', 'msg': 'Invalid letter'}}
                return

            player.letter_choice = letter
            group.submissions += 1
            print(f"[Game] Round {group.round_number}: Player {player.id_in_group} chose '{letter}' "
                  f"({group.submissions}/{group.num_humans} humans submitted)")

            advance = _check_and_advance(group)
            if advance:
                yield advance
            elif group.someone_timed_out:
                yield {p.id_in_group: {'type': 'advance'} for p in group.get_players()}
            else:
                yield {player.id_in_group: {'type': 'waiting'}}


class Results(Page):
    timeout_seconds = 45

    def is_displayed(player):
        if player.participant.vars.get('timed_out', False):
            return False
        if player.participant.vars.get('matching_timeout', False):
            return False
        if player.participant.vars.get('consent', False) != True:
            return False
        if player.participant.vars.get('attention_check_failed', False):
            return False
        # Hide Results if the group ended early
        if player.group.someone_timed_out:
            return False
        # Hide Results if LLM failed
        if player.group.llm_fallback:
            return False
        return True

    def vars_for_template(player):
        player.round_points = player.calculate_points()
        player.total_points = player.participant.vars.get('total_points', 0) + player.round_points
        player.participant.vars['total_points'] = player.total_points

        human_players = [p for p in player.group.get_players()
                         if not p.participant.vars.get('timed_out', False)]

        choices_with_labels = []
        for i, p in enumerate(human_players, start=1):
            choices_with_labels.append({
                'player_num': i,
                'player_type': 'Human',
                'choice': p.letter_choice or '?',
                'is_me': p.id_in_group == player.id_in_group,
            })

        raw = player.group.llm_choices_json
        num_llms = player.participant.vars.get('num_llms', 1)
        if raw and raw != 'PENDING':
            llm_choices = json.loads(raw)
        else:
            print(f"[Results] Round {player.round_number}: llm_choices_json is '{raw}' — showing '?' placeholders.")
            llm_choices = ['?'] * num_llms
        num_humans = len(human_players)
        for i, choice in enumerate(llm_choices, start=1):
            choices_with_labels.append({
                'player_num': num_humans + i,
                'player_type': 'AI',
                'choice': choice or '?',
                'is_me': False,
            })

        total_players = player.participant.vars.get('num_humans', 5) + player.participant.vars.get('num_llms', 1)

        all_matched = player.round_points > 0

        return {
            'my_choice': player.letter_choice,
            'my_points': player.round_points,
            'total_points': player.total_points,
            'point_values': player.get_available_point_values(),
            'is_new_scoring_round': player.round_number == 8,
            'round_number': player.round_number,
            'choices_with_labels': choices_with_labels,
            'total_players': total_players,
            'all_matched': all_matched,
        }


class TimeoutEnd(Page):
    def is_displayed(player):
        return (player.participant.vars.get('timed_out', False) and
                player.round_number == C.NUM_ROUNDS)


class MatchingTimeoutEnd(Page):
    def is_displayed(player):
        return (player.participant.vars.get('matching_timeout', False) and
                player.round_number == C.NUM_ROUNDS)


class FinalResults(Page):
    def is_displayed(player):
        return (not player.participant.vars.get('timed_out', False) and
                not player.participant.vars.get('matching_timeout', False) and
                player.participant.vars.get('consent', False) == True and
                not player.participant.vars.get('attention_check_failed', False) and
                player.round_number == C.NUM_ROUNDS)

    def vars_for_template(player):
        return {
            'total_rounds': C.NUM_ROUNDS,
            'completion_message': "Thank you for completing the experiment!",
        }


page_sequence = [
    WaitForPlayers,
    Name_Game,
    Results,
    TimeoutEnd,
    MatchingTimeoutEnd,
    FinalResults,
]
