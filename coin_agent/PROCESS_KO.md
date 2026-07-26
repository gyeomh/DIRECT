# coin_agent 전체 프로세스 & 프롬프트 정리 (한국어)

`graph_method/coin_questioner_spec.md` 스펙을 기반으로 구현한 `GraphQuestioner`의 전체 아키텍처와,
그 안에서 사용하는 세 가지 VLM(Qwen) 프롬프트를 단계별로 정리한 문서입니다.

---

## 1. 이 시스템이 하는 일

CoIN Challenge의 목표는 다음과 같습니다:

- 매 에피소드마다 **타겟 객체 하나**에 대한 **설명(description)** 텍스트가 주어집니다.
  (예: `"White cabinet standing against a light green wall"`)
- 그리고 여러 장의 **후보 이미지(candidate/distractor)**가 순서대로 주어집니다.
- 우리 모델(`Questioner`)은 각 후보 이미지를 보고 "이 이미지가 타겟 객체와 같은 물체인지" 를
  판단해야 합니다. 확신이 없으면 **오라클(Oracle)**에게 질문을 던져 답을 얻을 수 있습니다.
- 목표는: **질문을 최대한 적게 쓰면서, 정답률을 최대화**하는 것.

우리는 이 문제를, "매 순간 이미지 하나만 보고 새로 판단"하는 대신 —
**에피소드 전체에 걸쳐 타겟에 대해 알게 된 사실을 하나의 "노트"(믿음, belief)에 계속 누적**하고,
그 노트를 이용해 "이번 후보가 왜 타겟과 다른지/같은지"를 판단하는 방식으로 설계했습니다.

---

## 2. 설계 원칙 (스펙 §1)

1. **지속되는 것은 "타겟에 대한 믿음"이지, 이미지별 구조가 아니다.**
   후보 #1을 보면서 알게 된 사실은 후보 #2~#6에도 그대로 유효합니다. 그래서 에피소드당
   `TargetBelief` 하나만 만들고, 이미지별로는 가벼운 `ObservationFrame`만 새로 만듭니다.

2. **슬롯(구조화된 속성) 레이어는 최종 판단에 직접 관여하지 않는다.**
   슬롯 구조는 "어떤 질문을 할지 고르고, 충돌을 감지하는 데"만 씁니다. 최종 매치/노매치
   판단은 실제 픽셀 + 텍스트로 표현한 믿음을 보고 VLM이 직접 내립니다. 즉, 슬롯 추출이
   틀려도 그 대가는 "질문 하나 낭비"이지, "틀린 결론"이 아니어야 합니다.

3. **증거는 비대칭적이다.** 결정적인 충돌 하나만 있어도 "다른 물체"라고 확신할 수 있지만,
   여러 속성이 일치한다고 해서 "같은 물체"라고 확신할 수는 없습니다 (후보들은 서로 매우
   비슷하게 만들어져 있음). 그래서: 충돌이 발견되면 즉시 멈추고, 확실한 "매치"는 남은
   변별 요소가 없을 때만 내립니다.

4. **오라클에게는 예/아니오가 아니라 open-ended(육하원칙) 질문을 한다.**
   "네/아니오" 질문은 오라클이 무의식적으로 "네"라고 답하는 편향(acquiescence bias)에
   취약합니다. 잘못된 "네"가 가장 비싼 실수이므로, 항상 "무엇이 ~한가요?" 형태로 묻습니다.

5. **앞부분에 질문을 몰아서 한다(front-load).** 후보 #1에서 얻은 정보는 이후 모든 후보에
   재사용되므로, 초반에 여러 질문을 쓰고 후반에는 아끼는 것이 유리합니다.

6. **후보 이미지에서 읽을 수 없는 속성은 아무리 오라클이 잘 답해도 쓸모없다.**
   그래서 "이 후보 이미지에서 잘 안 보이는 속성"은 애초에 질문 후보에서 제외합니다.

---

## 3. 전체 데이터 흐름 (한 번의 `ask_or_conclude` 호출)

```
관찰값 observation = {image, answer}
      │
      ├─ 새로운 후보 이미지인가? ─예─▶ extract() [Qwen VLM] ─▶ ObservationFrame (이미지 해시로 캐싱)
      │                                                             │
      ├─ 오라클 답변이 있는가? ─▶ parse_oracle_answer() ─▶ belief.set_slot()
      │                                                             │
      ▼                                                             ▼
                 compare(frame, belief) ─▶ 결정적 충돌 있음? ─예─▶ 결론(False) 반환
                           │아니오
                           ▼
        select.candidate_pool() ─▶ 물어볼 게 없다(비어있음)? ─예─▶ 헤지(hedge)된 슬롯 때문인가?
                           │아니오                                    │예: adjudicate() 호출
                           │                                          │아니오: 결론(True) — 소거법
                           ▼
        budget.may_ask()? ─예─▶ select.top() ─▶ 질문(question) 반환
                           │아니오
                           ▼
                  adjudicate() [Qwen VLM] ─▶ 결론(bool) 반환
```

**핵심 포인트**
- `TargetBelief`(타겟에 대한 믿음/노트)는 에피소드 내내 유지되며, 한번 확정된 슬롯 값은
  절대 덮어쓰지 않습니다(monotonicity, §4). 설명 텍스트에서 왔든 오라클 답변에서 왔든 동일.
- `ObservationFrame`(이번 후보에서 읽어낸 값)은 이미지가 바뀔 때만 새로 만듭니다
  (`ask_or_conclude`가 같은 이미지에 대해 여러 번 호출돼도 재추출하지 않음 — idempotent).

---

## 4. 슬롯(Slot) 스키마와 티어(Tier) — `schema.py`

타겟 객체와 주변 맥락을 22개(인덱스 확장 포함)의 **고정된, 닫힌 슬롯 집합**으로 표현합니다.
(오픈월드 씬 그래프가 아니라, 딱 정해진 필드들만 다룹니다 — 설명 텍스트가 최대 2단계
관계 이상을 넘어가지 않기 때문입니다.)

| 티어 | 의미 |
|---|---|
| **A (결정적)** | 여기서 충돌이 나면 즉시 "다른 물체"로 결론 내릴 수 있음. 시점/조명에 안정적인 속성만 선정 (예: `obj.color_primary`, `room.type`, `ctx.above.material`) |
| **B (증거용)** | 최종 판단(adjudicate)에 참고는 되지만, 그 자체로는 절대 결정적이지 않음 (예: `obj.color_secondary`, `room.wall_color`) |
| **C (미사용)** | 일시적이거나 움직일 수 있는 것들 — 절대 결정적이지 않고, **질문도 하지 않음** (예: `obj.state`(열림/닫힘), `ctx.contains`(안에 든 물건)) |

**중요한 최적화 (스펙에는 없던 발견):**
- `obj.category`는 매 후보 이미지마다 다시 추출할 필요가 없습니다. 학습 데이터 528개
  후보 이미지 전수 조사 결과, **모든 후보가 타겟과 같은 카테고리**였습니다(불일치 0건).
  게다가 `env.py`가 `info["category"]`를 통해 카테고리를 직접 알려주므로(§0.6 유출과는
  다름 — 아래 6장 참고), VLM에게 굳이 다시 물어볼 이유가 없습니다.
- Tier C 슬롯들은 `compare.py`(충돌 판정)에도, `select.py`(질문 후보)에도,
  `adjudicate.py`(최종 판단 입력)에도 전혀 쓰이지 않습니다. 즉 **소비자가 하나도 없는
  값**이므로, 두 VLM 프롬프트(추출/설명파싱) 모두에서 아예 요청하지 않습니다
  (`schema.queryable_slot_keys()` 참고).

---

## 5. 값 정규화와 유사도 — `canon.py`

VLM/오라클이 자유 텍스트로 답하기 때문에("navy blue", "네이비", "짙은 파란색" 등),
같은 뜻의 표현들을 하나의 **정규값(canonical value)**으로 매핑합니다
(예: `"navy blue"`, `"dark navy"` → `navy`).

그리고 두 정규값 사이의 관계를 세 가지로 분류합니다:

- **SAME**: 정규화 후 완전히 같음
- **NEAR**: 조명/촬영각/압축에 따라 같은 것일 수 있음 (`NAVY ~ DARK_BLUE ~ BLUE`) — 절대
  결정적이지 않음, "약한 충돌(WEAK_CONFLICT)"로만 취급
- **FAR**: 서로 완전히 다름 (`NAVY / WHITE`, `BRASS / CHROME`) — 이것만 결정적 충돌 가능

이 표를 문자열 유사도로 자동 추론하지 않고 **명시적인 표로 직접 관리**합니다 — 잘못된
FAR 판정 하나가 바로 "틀린 결론"으로 이어지기 때문입니다.

---

## 6. `info["category"]` — 스펙에 없던 중요한 발견

`env.py:135`를 직접 읽어보니, `reset()`이 호출될 때마다 `--description-type` 설정과
**상관없이 항상** `info["category"]`를 넣어줍니다 (예: `"Cabinet"`).

이것은 스펙 §0.6에서 금지하는 `info["task_image"]` 유출(타겟 이미지 자체)과는 다릅니다 —
카테고리 이름은 모든 description 타입에 대해 주최측이 의도적으로 제공하는 값입니다. 그래서:

- `obj.category` 슬롯은 이 값으로 **100% 신뢰도로 즉시 채웁니다** (텍스트에서 추측할 필요 없음).
- 이 값은 오라클에게 물어볼 질문의 명사구(noun phrase)로도 사용됩니다
  (예: `"the cabinet"` → *"What is the primary color of the cabinet?"*).

---

## 7. 세 개의 프롬프트

이 시스템은 Qwen VLM에게 정확히 **세 번의 서로 다른 목적**으로 요청을 보냅니다.
아래는 각 프롬프트의 목적, 실제 사용된 전체 프롬프트 텍스트, 그리고 설계 이유입니다.

### 7-1. `DESCRIPTION_PARSE_PROMPT` — 설명 텍스트 → 초기 노트

**언제 호출되는가:** 에피소드 시작 시, `GraphQuestioner.__init__` 안에서 **딱 한 번**,
아직 어떤 후보 이미지도 보기 전에 호출됩니다. **이미지 없이 텍스트만** 넣는 호출입니다
(같은 Qwen 모델을 텍스트 전용으로 사용).

**목적:** `target_description`이 이미 말해주고 있는 사실들(색, 재질, 주변 물체와의 관계 등)을
미리 "노트"(`TargetBelief`)에 채워 넣습니다. 이렇게 해야:
- `select.py`가 "설명에 이미 나온 속성은 다시 묻지 않는다"는 필터를 실제로 작동시킬 수 있고,
- `budget.py`의 "설명이 이미 언급 안 한 Tier-A 속성 개수"(`ambiguity_allowance`)가 정확해집니다.

**실제 프롬프트 텍스트** (`{description}`, `{not_mentioned}`, `{schema_keys}`는 호출 시 실제
값으로 채워짐):

```text
You are extracting structured facts from one short description of an object and its immediate surroundings.

Description: "{description}"

The object this sentence describes IS the target object — extract facts ABOUT it and about what is spatially around it. Anything else the sentence names (a picture, a doorway, a wall) is surrounding context, not the target itself.

Extract ONLY facts explicitly stated or directly, unambiguously implied by this exact sentence. Never guess, never fill in a plausible-sounding default, never use outside knowledge about what this kind of object usually looks like. If the description does not mention a field, its value must be exactly "{not_mentioned}".

Rules (field names below are exactly the JSON keys you must use — see the schema at the bottom):
- "X and Y <object>" (e.g. "white and multicolored clock") -> the first color is obj.color_primary, the second is obj.color_secondary.
- Object-name fields (ctx.above.object, ctx.support.object, ctx.adjacent[0].object, room.notable_appliance) must be a single bare noun with no adjectives — "doorway", not "open doorway"; "picture", not "black framed picture". Put color separately in the matching *.color field. Drop any adjective that doesn't fit one of these fields (e.g. "open", "framed", "faceted") rather than folding it into the noun.
- "beneath/under/below X" -> the object is BELOW X, so X is ctx.above.object (plus ctx.above.material/ctx.above.color if X's material/color is also given).
- "standing on/set into/resting on/hanging on X" -> X is ctx.support.object.
- "next to/beside/against X" -> X is ctx.adjacent[0].object (plus ctx.adjacent[0].color if X's color is given).
- "with <material> accents/trim" (e.g. "red tile accents") -> that material is obj.material, and its color is obj.color_secondary.
- A wall's color mentioned anywhere -> room.wall_color. A floor's color/material -> room.floor_color / room.floor_material.
- Loose or movable items (stuffed animals, plush toys, towels) are contents, not context — there is no field for them; leave every other field "{not_mentioned}" if this is all the sentence says.

Return ONLY strict JSON, no other text, with exactly these keys and no others:
{schema_keys}
```

**왜 이렇게 썼는가 (실제 데이터 예시로 검증한 규칙들):**
- 실제 description 예시들 (`"Clock hanging on a wall next to a framed picture"`,
  `"White shower with red tile accents next to an open doorway"` 등)을 직접 분석해서 규칙을
  뽑았습니다:
  - `"X and Y <물체>"` (예: *"white and multicolored clock"*) → 첫 색은 주색상, 둘째 색은 보조색상
  - `"~ 위에 걸려있다/붙어있다"` → 지지대(`ctx.support.object`)
  - `"~ 아래에 있다"` → 위쪽 물체(`ctx.above.object`)
  - `"~ 옆에 있다"` → 인접 물체(`ctx.adjacent[0].object`)
  - `"~ 장식/트림이 있다"`(예: *"red tile accents"*) → 재질(`obj.material`) + 보조색상
- **"바른 명사만 쓰라"는 규칙이 특히 중요합니다.** 예를 들어 설명에서는
  `"open doorway"`라고 했지만 나중에 이미지 추출 프롬프트가 같은 물체를 `"doorway"`라고만
  답하면, `compare.py`는 자유 텍스트 슬롯의 불일치를 **FAR(결정적 충돌)**로 처리하도록
  설계했기 때문에 — 단순 표현 차이(paraphrase)만으로 **가짜 충돌**이 생겨 틀린 결론이
  나올 뻔했습니다. 그래서 두 프롬프트 모두 "형용사 빼고 명사만" 규칙을 동일하게 적용합니다.
- `obj.category`는 이 프롬프트에서 아예 요청하지 않습니다 — `info["category"]`로 이미
  100% 확실하게 알고 있기 때문입니다 (6장 참고).

---

### 7-2. `EXTRACTION_PROMPT` — 후보 이미지 → 관찰값

**언제 호출되는가:** 새로운 후보 이미지를 처음 볼 때마다 (같은 이미지에 대해
`ask_or_conclude`가 여러 번 불려도 재호출하지 않음 — 이미지 해시로 캐싱).

**목적:** 지금 보고 있는 후보 이미지에서 각 슬롯 값을 관찰합니다. **설명(description)에
대해 전혀 모르는 상태로** 관찰해야 합니다 — 그래야 "설명과 맞춰보려는" 무의식적 편향
없이, 순수하게 "이 이미지에 실제로 뭐가 보이는가"만 답하게 됩니다. (이 편향 없음이
`compare.py`가 정확히 작동하기 위한 전제조건입니다.)

**실제 프롬프트 텍스트** (이미지와 함께 전송; `{unclear}`, `{schema_json}`는 실제 값으로 채워짐):

```text
Look closely at the image and report exactly what you can observe about the single main object in it and its immediate surroundings.

Describe only what is visible in THIS image. Do not reference any other image, any description, or what this kind of object usually looks like elsewhere — report only what you can actually see here.

For every field below, provide:
- "value": a short, literal, lowercase phrase for what you observe, or exactly "{unclear}" if you cannot determine it from this image.
- "visibility": "clear" if you can confidently observe it, "partial" if it is partially visible, occluded, or ambiguous, "not_visible" if it is not visible in this image at all.

Rules:
- Ignore compression artifacts, digital noise, rendering glitches, or watermarks entirely — never report them as real content.
- Object-name fields (ctx.above.object, ctx.support.object, ctx.adjacent[i].object, room.notable_appliance) must be a single bare noun with no adjectives — "doorway", not "open doorway"; "picture", not "black framed picture". Put color/material in the matching separate field instead of folding it into the noun.
- If a relation genuinely doesn't apply to this scene (e.g. there is no adjacent object at all), set its value to "{unclear}" and its visibility to "not_visible" — do not invent one.

Return ONLY strict JSON, no other text, with exactly these keys and no others, each an object with "value" and "visibility":
{schema_json}
```

**설계 포인트:**
- 필수 요구사항 4가지 (스펙 §5.1) 를 모두 만족합니다:
  1. `response_schema()`와 정확히 일치하는 JSON 구조
  2. 모든 필드에 명시적인 `"unclear"`(모르겠음) 선택지
  3. 필드별 자기신고 가시성(`visibility`): `clear` / `partial` / `not_visible`
  4. 압축 아티팩트/노이즈를 절대 실제 내용으로 보고하지 말라는 명시적 지시
- **설명-블라인드(description-blind):** "target"이라는 단어도, 설명 텍스트에 대한 언급도
  전혀 없습니다. `ADJUDICATION_PROMPT`와 정반대 성격입니다 (아래 참고).
- 신뢰도(confidence) 계산: 원래 스펙은 "평균 토큰 로그확률 × visibility"를 곱하라고
  했지만, 아직 실제 로그확률을 슬라이싱하는 로직을 구현하지 않았습니다. **여기서 실제 버그를
  하나 발견하고 고쳤습니다:** 로그확률 자리에 임시로 중립값 0.5를 곱해 넣었더니,
  `"clear"`(확실히 보임)라고 답한 값조차 신뢰도가 `0.5 × 1.0 = 0.5`가 되어버려서 —
  기본 임계값(`tau_obs = 0.80`)보다 낮아 **모든 슬롯이 항상 "신뢰도 부족"으로 걸러지는**
  치명적인 버그였습니다. 지금은 신뢰도를 **자기신고 visibility 값 하나만으로** 계산하도록
  고쳤고, 실제 로그확률 기반 보정은 나중에 v2로 미뤘습니다 (거짓으로 채워 넣지 않음).

---

### 7-3. `ADJUDICATION_PROMPT` — 최종 매치/노매치 판단

**언제 호출되는가:** 두 가지 경우에만 호출되는 "마지막 수단"입니다.
1. 질문 예산(budget)을 다 쓴 뒤에도 아직 결론을 못 냈을 때
2. 물어볼 게 더 없는데(pool이 비었는데), 그게 "진짜 확실해서"가 아니라 유일하게 남은
   슬롯이 **애매하게(hedge) 답변된 경우** (예: *"잘 모르겠지만 아마 화강암인 것 같아요"*)
   — 이 경우 무작정 "매치"로 결론 내리면 안 되므로 adjudicate를 호출하도록 설계했습니다.

**목적:** 슬롯 구조(frame)는 절대 넘기지 않고, **실제 이미지 픽셀 + 지금까지 누적된 노트
텍스트**만 보고 최종 판단을 내리게 합니다. 이렇게 하면 슬롯 추출 과정의 실수가 "틀린 결론"이
아니라 "질문 하나 낭비"로만 끝난다는 설계 원칙(§1-2)을 지킬 수 있습니다.

**실제 프롬프트 텍스트** (이미지와 함께 전송; `{description}`, `{belief_text}`,
`{qa_history}`는 실제 값으로 채워짐):

```text
You are deciding whether the object shown in this image is the same specific object as a target object described below, which you have not seen directly.

Target description: "{description}"

Additional confirmed facts about the target (from the description, or from questions already asked and answered by an oracle who has seen the target directly):
{belief_text}

Questions already asked and their answers:
{qa_history}

Look at the image above. Compare what you actually see in it against the description and the confirmed facts. Candidates are often near-duplicates that differ only in a few details (color, material, a nearby object, the room) — a single concrete, confirmed mismatch on any of these means this is NOT the same object, even if everything else matches. Ignore compression artifacts, digital noise, or rendering glitches entirely — never treat them as a real difference.

Provide your reasoning, then a score:
- 2 if you are confident this is the same specific object as the target.
- 0 if you are confident this is NOT the same object (something concretely conflicts).
- 1 if you are genuinely unsure either way.

Strictly follow this output format: <motivation>your reasoning here, under 60 words, do NOT use double quotes</motivation><score>0, 1, or 2</score>
```

**설계 포인트:**
- `EXTRACTION_PROMPT`와 정반대로, 이 프롬프트는 **"target"이라는 개념을 자유롭게 언급**합니다
  — 이 호출 자체의 목적이 "후보를 타겟 개념과 비교하는 것"이기 때문입니다.
- 출력 형식은 주최측이 제공한 베이스라인 예시 프롬프트(`Questioner.py`의
  `QUESTIONER_EXAMPLE_PROMPT`)의 `<motivation>/<score>` 태그 형식을 그대로 재사용했습니다
  (단, `<question>` 태그는 뺐습니다 — 이 호출은 더 질문할지 말지를 정하는 게 아니라,
  이미 예산이 끝나서 마지막으로 결론만 내리는 호출이기 때문입니다).
- 점수(score) → 결론(conclusion) 매핑: `2 → True(매치)`, `0 → False(노매치)`,
  `1(애매함) → False`. 한 에피소드당 정답(match=True) 후보는 정확히 1개뿐이므로, 후보 하나가
  맞을 기본 확률은 대략 1/6 — 그래서 "애매하면 동전던지기"보다 "애매하면 아니라고 본다"가
  더 낫다는 판단입니다.
- **여기서도 실제 버그 하나를 발견하고 고쳤습니다:** `select.py`는 같은 영역(region)의 슬롯
  2개를 질문 하나로 묶어서 물어볼 수 있는데(예: *"주 색상이 뭐고, 손잡이 마감재는 뭔가요?"*),
  이때 `questioner.py`가 그 질문을 **묶인 슬롯 개수만큼(2번)** `belief.asked` 리스트에
  똑같은 질문 텍스트로 기록해 버립니다. 그래서 `qa_history_text()`가 이 히스토리를 그대로
  나열하면 **똑같은 질문-답변 쌍이 두 번 찍히는** 문제가 있었습니다. 질문 텍스트 기준으로
  중복 제거하도록 고쳤습니다.

---

## 8. 충돌 판정 로직 — `compare.py`

관찰값(frame)과 노트(belief)에 동시에 존재하는 슬롯마다 다음 다섯 가지 중 하나로 분류합니다:

- `MATCH`: 정규값이 완전히 같음
- `CONFLICT` (결정적): **다음 네 조건을 모두 만족해야만** 성립
  1. Tier A 슬롯이어야 함
  2. 노트 쪽 값이 "hedge(애매함)"이 아니라 확실히 확정된 값이어야 함
  3. 관찰값의 신뢰도가 `tau_obs`(기본 0.80) 이상이어야 함
  4. 두 값의 관계가 `FAR`이어야 함
- `WEAK_CONFLICT`: 위 조건을 만족하지 못하는 모든 불일치 (그 자체로는 절대 결론에 영향 없음.
  단, 설정으로 "약한 충돌이 N개 이상 쌓이면 결정적으로 취급"하는 옵션이 있으나 기본은 꺼둠)
- `UNKNOWN` / `INCOMPARABLE`: 비교 불가

**자유 텍스트 슬롯(예: `ctx.above.object`)에 대한 설계 결정:** 이런 슬롯은 동의어 표(NEAR/FAR
표)가 없으므로, 원래는 "정확히 같은 문자열이 아니면 NEAR" 취급했는데, 이렇게 하면 Tier A로
지정된 자유 텍스트 슬롯들(예: 무엇 위에 있는지, 무엇 위에 놓여있는지)이 절대 결정적 충돌을
낼 수 없게 되어버립니다. Tier A로 지정한 의도(시점에 안정적이라 결정적으로 써도 된다)와
모순되므로, 자유 텍스트 슬롯은 "다르면 FAR"로 처리하도록 바꿨습니다 — 대신 두 프롬프트 모두
"형용사 없이 명사만" 규칙을 강제해서, 단순 표현 차이 때문에 가짜 충돌이 나는 위험을 줄였습니다.

---

## 9. 질문 선택 — `select.py`

1. **후보 풀(pool) 구성**: 다음을 모두 만족하는 슬롯만 질문 후보가 됩니다
   - Tier A 또는 B (Tier C는 절대 제외)
   - 노트에서 아직 `unknown`(모름) 상태 — 오라클은 결정적(deterministic)이므로 같은 걸
     두 번 묻지 않음
   - 이번 후보 이미지에서 신뢰도 있게 읽힌 값이어야 함 (`tau_obs` 이상) — 이미지에서 안 보이면
     오라클이 아무리 잘 답해도 쓸모없음
   - 설명 텍스트에 이미 나와 있지 않은 것 (7-1의 `DESCRIPTION_PARSE_PROMPT`가 채운 슬롯 제외)

2. **점수화**: `frame.confidence × 변별력(disc) × 티어가중치 × 안정성(stability)` — 변별력은
   "이 카테고리에서 이 값이 얼마나 드문가"를 학습 데이터로 추정한 값 (`priors.py`,
   `scripts/build_priors.py`).

3. **번들링(bundling)**: 같은 영역(region)의 슬롯 최대 2개를 질문 하나로 묶을 수 있음
   (예: "주 색상 + 손잡이 마감재"). 다른 영역끼리는 절대 묶지 않음 — 오라클 답변이 15단어
   이하로 제한되어 있어서, 서로 무관한 두 정보를 한 문장에 다 담기 어렵기 때문입니다.

4. **앞부분 몰아넣기(front-load)**: 첫 번째 후보에서는 점수 순이 아니라, **서로 다른 영역별로
   골고루** 하나씩 뽑습니다 (예: 객체 자체 → 위쪽 맥락 → 방 전체). 방 정보 하나가 확정되면
   이후 모든 후보에 대해 "전혀 다른 방인 후보"를 공짜로 걸러낼 수 있기 때문입니다.

---

## 10. 예산 관리 — `budget.py`

- 전체 스텝 한도: 60 스텝, 전체 시간 한도: 600초 (이 두 값은 `env.py`에서 읽은 사실 그대로).
- 후보가 몇 개 남았는지 정확히 알 수 없으므로, "남은 후보 수"를 보수적으로 가정(기본 8개,
  실제 예시는 6개)하고, 후보마다 결론 내리는 데 최소 1스텝씩은 남겨둡니다(reserve).
- 후보당 질문 개수 상한은: (남은 스텝 수 ÷ 남은 후보 수) × 앞부분가중치, 그리고
  "설명이 아직 언급하지 않은 Tier-A 슬롯 개수"와 하드캡(기본 6) 중 최솟값으로 정해집니다.
  → 이 덕분에 `category`(설명이 거의 정보를 안 줌)일 때는 질문을 많이, `color_context_feature`
  (설명이 정보를 많이 줌)일 때는 질문을 적게 쓰도록 자동으로 조절됩니다.
- 소프트 타임아웃(60%) 이후엔 새 질문을 멈추고, 하드 타임아웃(85%) 이후엔 판단 호출 자체도
  건너뛰고 바로 "노매치"로 처리합니다 (그 시점엔 VLM 호출할 시간조차 아까움).

---

## 11. 스펙에 없던 것 중 추가로 발견한 사실들

1. **`info["category"]`가 항상 제공됨** (6장 참고) — `obj.category`를 텍스트에서 추측할
   필요가 없다는 것을 알려줌.
2. **학습 데이터 167개 에피소드 전수 조사 결과, `match=True` 후보가 항상 리스트의 마지막**
   이었습니다 (`scripts/analyze_episodes.py`로 직접 확인). 스펙이 경고한 그대로였지만, 실제
   데이터로 재확인한 것 — 그리고 이 사실에 **절대 의존하면 안 됩니다** (홀드아웃 세트는
   순서가 다시 섞여 있다고 주최측이 명시함).
3. **모든 후보 이미지가 타겟과 같은 카테고리** (528개 전수 조사, 불일치 0건) — 그래서
   `obj.category`를 이미지에서 재추출할 필요가 없음 (4장 참고).
4. **`env.MockOracle.ask()`가 `self` 파라미터 없이 정의되어 있어서, 그대로 호출하면
   `TypeError`가 나는 실제 버그**를 발견했습니다 (주최측 코드 자체의 버그, 스펙에 나온
   3가지 알려진 이슈와는 별개). 우리 테스트에서는 이 클래스를 직접 쓰지 않고 흉내만 냅니다.

---

## 12. 발견하고 고친 실제 버그들 (요약)

| # | 위치 | 문제 | 고친 방법 |
|---|---|---|---|
| 1 | `questioner.py` | 이미지 추출(`extract()`)이 실패하면 "모든 슬롯 모름" 상태가 되는데, 이게 우연히 "물어볼 게 다 떨어짐(pool 비어있음)"과 똑같은 모양이 되어버려서, 추출 실패를 "확실한 매치"로 잘못 결론 내릴 뻔했음 | 추출 실패 여부를 별도 플래그로 추적해서, 그 경우엔 무조건 `adjudicate()`로 보내도록 분리 |
| 2 | `parse.py` | 오라클 답변에서 "헤지(hedge, 애매함)"와 "정보없음"을 판단하는 단어 목록이 서로 겹쳐서("not sure"가 둘 다에 있음), *"잘 모르겠지만 아마 화강암"* 같은 답이 값 자체를 버리고 "정보없음"으로 처리되는 버그 | "정보없음" 판정을 없애고, 값 추출에 성공했는지 여부만으로 분기하도록 단순화 |
| 3 | `compare.py` | 자유 텍스트 슬롯(예: 위쪽 물체 이름)의 불일치가 항상 `NEAR`(비결정적)로만 처리되어, Tier A로 지정한 의도와 모순 | 자유 텍스트 슬롯은 "다르면 `FAR`"로 변경 (대신 두 프롬프트에 "형용사 없이 명사만" 규칙 추가) |
| 4 | `select.py` / `questioner.py` | 유일하게 남은 변별 슬롯이 "헤지된 답변"으로만 채워져 있을 때, pool이 비었다는 이유로 무작정 "매치"로 결론 내릴 뻔함 | `select.has_hedged_discriminative_slot()`을 추가해서, 이 경우엔 `adjudicate()`를 거치도록 분리 |
| 5 | `extract.py` | 신뢰도 계산 시 아직 구현 안 된 로그확률 자리에 중립값(0.5)을 임시로 곱해 넣어서, `"clear"`(확실히 보임)라고 답한 슬롯도 신뢰도가 기본 임계값(0.80)보다 낮아지는 치명적인 버그 | 로그확률 슬라이싱을 실제로 구현하기 전까지는, 신뢰도를 자기신고 visibility 값 하나로만 계산 |
| 6 | `adjudicate.py` | 번들 질문(슬롯 2개를 한 질문으로 묶음)이 `belief.asked`에 슬롯 개수만큼 중복 기록되어, 최종 판단 프롬프트의 질문-답변 히스토리에 같은 문답이 두 번 나옴 | 질문 텍스트 기준으로 중복 제거 |

---

## 13. 현재 테스트 현황

- 총 **90개 테스트**, 전부 통과 (`pytest coin_agent/tests`).
- 순수 로직(스키마, 정규화, 비교, 선택, 예산, 상태)은 네트워크 없이 완전히 테스트됨.
- `extract()` / `adjudicate()`는 monkeypatch로도 검증하고, **아래 14장에서처럼 실제 모델로도
  검증**함.

---

## 14. 실제 모델로 전체 파이프라인 실행 검증 (라이브 테스트)

`Qwen/Qwen3-VL-30B-A3B-Instruct`를 vllm으로 GPU 2 한 대에만 띄워서, 세 프롬프트 전부와
`GraphQuestioner` 전체 루프를 실제로 돌려봤습니다.

**환경 구성에서 겪은 문제:** 이 서버의 NVIDIA 드라이버가 CUDA 12.8까지만 지원하는데, 최신
vllm(0.26.0)은 기본적으로 CUDA 13 전용 torch를 요구해서 바로 실패했습니다
(`RuntimeError: The NVIDIA driver on your system is too old`). torch만 따로 cu128 버전으로
바꿔치기하니 이번엔 vllm 자체의 컴파일된 확장 모듈이 `libcudart.so.13`을 요구하며 깨졌습니다
(vllm의 사전 컴파일된 커널 자체가 CUDA 13 대상이라, torch만 바꾼다고 해결되지 않음). 해결책:
`vllm==0.15.0`으로 내려서 설치 — 이 버전이 고정하는 `torch==2.9.1`은 PyPI 기본 빌드 자체가
`+cu128`이라, 드라이버와 자연스럽게 맞아떨어짐. Qwen3-VL-MoE 아키텍처 지원은 이 버전에도 이미
있었습니다.

**14-1. 프롬프트 3개 개별 검증 (실제 모델 응답)**

- `DESCRIPTION_PARSE_PROMPT`: 실제 예시 문장들("White cabinet with plush toys under a floral
  painting", "White and multicolored clock next to a black framed picture" 등)을 넣어보니
  `ctx.above.object: painting`, `obj.color_secondary: multicolor` 등 규칙대로 정확히 추출됨.
- `EXTRACTION_PROMPT`: 실제 후보 이미지(캐비닛 사진)에 대해 `obj.color_primary: white`,
  `ctx.above.object: picture`, `room.wall_color: green` 등, 이미 알고 있던 정답과 정확히 일치.
- `ADJUDICATION_PROMPT`: 일치하는 노트를 주면 `True`+타당한 근거, 충돌하는 노트(예: 남색
  vs 실제로는 흰색)를 주면 `False`+정확한 충돌 지점을 짚어내는 근거를 반환함.

**14-2. `env.QAEnv` + `GraphQuestioner`로 실제 에피소드 14개 실행**

Gemini API 키가 없으므로, 오라클도 우리 자신의 Qwen 모델(같은 서버)로 대체해서 실행 —
README가 명시적으로 허용하는 방식입니다. 결과: **14개 에피소드 전부 크래시 없이, 항상 유효한
행동(question/conclusion 중 정확히 하나)을 반환하며 종료**. 다만 정확도는:
- 후보가 1개뿐인 에피소드(항상 정답 후보 하나뿐이라 사실상 쉬운 케이스)는 6가지 설명 타입
  전부에서 6/6 정답.
- 후보가 5개인 더 어려운 에피소드는 6번 시도 모두 완전히 맞히지 못함 — 아래 15장의 미해결
  설계 이슈로 이어짐.

**14-3. 라이브 테스트로 발견하고 고친 실제 버그 5개**

| # | 위치 | 문제 | 고친 방법 |
|---|---|---|---|
| 1 | `canon.py` | 색상 동의어 표에 "light blue", "dark grey"처럼 특정 조합만 있고, "light green", "multicolored" 같은 다른 조합은 전혀 없어서 모델이 정확히 답해도 매칭 실패 | 모든 조합을 일일이 나열하는 대신, "light/dark/pale/..." 같은 수식어를 떼고 기본 색상으로 재시도하는 로직 추가 |
| 2 | `llm.py` | `want_logprobs=True`일 때 openai SDK가 반환하는 `TopLogprob` 객체를 그대로 JSON 캐시에 저장하려다 크래시 — 실제 서버를 쓰는 순간 모든 추출 호출이 실패할 뻔함. 게다가 크래시 중간에 손상된 캐시 파일이 남아서, 이후 같은 키로는 영원히 실패하는 2차 문제까지 있었음 | 로그확률을 일반 dict로 변환해서 저장, 캐시 파일 쓰기를 원자적으로(임시 파일 후 rename) 변경 |
| 3 | **(가장 영향이 컸던 버그)** `compare.py` | 자유 텍스트 슬롯(예: 위쪽 물체 이름)의 불일치를 결정적 충돌(`FAR`)로 처리했었는데, 실제로 테스트해보니 곧바로 역효과가 남: `extract()`는 규칙대로 "picture"(명사만)라고 정확히 답했는데, 오라클은 같은 질문에 "소녀가 그네를 타는 그림, 꽃과 나비가 있는"처럼 장황하게 답해서 — 표현 차이만으로 진짜 정답을 오답으로 뒤집어버림 (첫 테스트 에피소드에서 바로 발생) | 자유 텍스트 슬롯은 다시 `NEAR`(비결정적)로 되돌림 — "가정하지 말고 검증하라"는 스펙 §8의 철학 그대로, 가설을 실제로 테스트해보고 틀렸다는 걸 확인한 뒤 되돌린 것. 오라클의 장황한 답변을 핵심 명사만 남기도록 정리하는 로직도 함께 개선 |
| 4 | `extract.py` | 신뢰도 계산 시 아직 구현 안 된 로그확률 자리에 중립값(0.5)을 임시로 곱해서, "clear"라고 답한 슬롯도 신뢰도가 기본 임계값(0.80)보다 낮아지는 버그 (12장의 버그 5와 동일) | 로그확률 슬라이싱 구현 전까지는 신뢰도를 visibility 값 하나로만 계산 |
| 5 | `adjudicate.py` | 번들 질문의 문답이 최종 판단 프롬프트에 중복 출력 (12장의 버그 6과 동일) | 질문 텍스트 기준 중복 제거 |

---

## 15. 아직 풀지 못한 설계 이슈 (다음에 논의할 것)

후보가 5개인 에피소드에서 반복적으로 관찰된 패턴: `select.candidate_pool()`이 비어서
"더 확인할 게 없으니 매치로 결론"(소거법, elimination) 처리되는데, 실제로는 정답이 아닌
후보에서 이 결론이 나온 것으로 보입니다.

**가설:** pool이 비는 이유가 두 가지로 갈릴 수 있는데, 지금은 이 둘을 구분하지 못합니다.
1. (스펙이 의도한 대로) 확인 가능한 모든 것을 다 확인했고 전부 일치함 → 매치가 맞음.
2. 이번 후보의 사진 자체가 정보가 부족해서(신뢰도 낮음), 남은 슬롯들을 애초에 확인할 방법이
   없었을 뿐 → 매치를 "확인"한 게 아니라 그냥 "확인을 못 한" 것.

12장의 버그 4번(헤지된 답변 때문에 pool이 비었을 때 무작정 매치로 결론 내리던 문제)과 근본적으로
같은 모양의 문제이지만, 이번엔 원인이 "헤지된 답변"이 아니라 "이번 후보 이미지의 관찰 신뢰도
자체가 낮음"이라는 점이 다릅니다. 이건 아직 고치지 않았습니다 — "이번 후보가 그냥 정보가
부족했다"와 "정말로 다 확인해서 일치했다"를 어떻게 구분할지는 설계 판단이 필요해서, 임의로
고치기보다 다음에 함께 논의하기로 했습니다.

---

## 16. 아직 안 한 것 / 다음 단계

- `scripts/build_priors.py` (변별력 표 구축), `scripts/run_eval.py` (실제 평가 실행),
  `scripts/ablate.py` (단계별 성능 비교) — 구조는 다 짜여 있고 이제 살아있는 모델도 있으니
  바로 돌려볼 수 있음.
- 15장의 "소거법 오탐" 설계 이슈 해결.
- 이 작업물은 지금 로컬 git 워크트리 브랜치(`worktree-coin-agent-arch`)에만 커밋되어 있고,
  아직 실제 `~/coin_challenge` 작업 폴더나 원격 저장소에는 반영되지 않았습니다.
- vllm 서버가 현재 GPU 2에서 계속 실행 중입니다 (필요 없으면 종료 가능).
