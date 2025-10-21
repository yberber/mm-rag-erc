from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 1. Define the System Prompt (the instructions for the model)
system_template = """Now you are an expert in dialogue emotion recognition.
Based on the provided conversation context and audio features, please select the emotional label of the target utterance from the candidate list.
Please output only the selected emotion label and make no explanation."""

# 2. Define the Human/User Prompt (the data for the task)
human_template = """Example(s):
{demonstrations}

The following conversation noted between '### ###' involves several speakers.
###
{history}
###

Based on the above conversation context, please select the emotional label of < "{speaker_id}" : "{utterance}" >, said with < {audio_features} >, from < {candidate_emotions} >."""

# 3. Create the ChatPromptTemplate
DEFAULT_LLAMA3_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
)
DEFAULT_LLAMA3_PROMPT.name = "DEFAULT_LLAMA3_PROMPT"





from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# System message (the instructions)
gemini_system = """You are a highly skilled AI expert in multimodal emotion recognition. Your task is to analyze the provided conversation, a specific target utterance, and its audio features to determine the speaker's emotion from a given list of options."""

# User message (the data)
gemini_human = """<task_definition>
Analyze the data below to identify the emotion of the speaker in the target utterance.
</task_definition>

<examples>
{demonstrations}
</examples>

<conversation_context>
<history>
{history}
</history>
<target_utterance>
  <speaker>{speaker_id}</speaker>
  <text>{utterance}</text>
  <audio_features>{audio_features}</audio_features>
</target_utterance>
</conversation_context>

<emotion_options>
{candidate_emotions}
</emotion_options>

Based on all the provided information, what is the emotion of the target utterance? Provide your answer inside <emotion> XML tags. Do not include any other text or explanation."""

# Assistant prefix (to guide the output format)
# The original template ended with "<emotion>", which is an assistant prefix.
gemini_ai = "<emotion>"

# Corrected ChatPromptTemplate for Llama 3.1
GEMINI_LLAMA3_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(gemini_system),
    HumanMessagePromptTemplate.from_template(gemini_human),
    AIMessagePromptTemplate.from_template(gemini_ai)
])
GEMINI_LLAMA3_PROMPT.name = "GEMINI_LLAMA3_PROMPT"



# System message (the instructions and rules)
claude_system = """You are an expert in sentiment and emotion analysis. Your task is to identify the emotion expressed in a specific utterance based on conversation context and audio features.

## Instructions:
1. Consider the conversation context and how it relates to this utterance
2. Take into account the audio features provided
3. Select the most appropriate emotion label from the available options
4. Output ONLY the emotion label with no additional text or explanation"""

# User message (the data)
claude_human = """## Examples:
{demonstrations}

## Conversation History:
{history}

## Task:
Analyze the following utterance and identify its emotion:
- Speaker: {speaker_id}
- Utterance: "{utterance}"
- Audio Features: {audio_features}

## Available Emotion Labels:
{candidate_emotions}"""

# Assistant prefix (to guide the output)
claude_ai = "Emotion Label:"

# Corrected ChatPromptTemplate for Llama 3.1
CLAUDE_LLAMA3_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(claude_system),
    HumanMessagePromptTemplate.from_template(claude_human),
    AIMessagePromptTemplate.from_template(claude_ai)
])
GEMINI_LLAMA3_PROMPT.name = "GEMINI_LLAMA3_PROMPT"





# System message (all instructions, rules, and output format)
gpt5_system = """You are a careful emotion classifier.

TASK
- Pick exactly ONE label from the allowed set.
- Do NOT invent new labels.
- Answer with ONLY the label text, nothing else.

ALLOWED_LABELS
<{candidate_emotions}>

DECISION POLICY
- Prioritize the current utterance first, then conversation history, then audio cues.
- If cues conflict, prefer the utterance's literal meaning.
- If uncertainty remains between two or more labels, choose the most non-committal one present in ALLOWED_LABELS
  (e.g., "neutral" or "other"/"unknown" if provided).

OUTPUT
Return ONLY one label from ALLOWED_LABELS."""

# User message (the specific data for this one task)
gpt5_human = """CONTEXT
Few-shot examples:
{demonstrations}

Conversation (delimited by ###):
###
{history}
###

TARGET
Speaker: "{speaker_id}"
Utterance: "{utterance}"
Audio descriptors: {audio_features}"""

# Corrected ChatPromptTemplate for Llama 3.1
# No AI prefix is needed as the model is just expected to output the label directly.
GPT5_LLAMA3_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(gpt5_system),
    HumanMessagePromptTemplate.from_template(gpt5_human)
])

GPT5_LLAMA3_PROMPT.name = "GPT5_LLAMA3_PROMPT"