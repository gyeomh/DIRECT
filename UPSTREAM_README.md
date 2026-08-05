# Explanation

# Installation

Run
> source scripts/install.sh

This will create a new local environment will all the required packages.

Next, download the required images
> mkdir images && hf download --repo-type dataset e-zorzi/images_coin_challenge --local-dir images

If you need `vllm` (see later), run

> source scripts/install_vllm.sh

This will create a *second* environment with vllm installed (this to avoid env pollution with the first).

# Oracle

To train/test your questioner, you need an Oracle. You can implement it however you want, but it needs to inherit from the class `OracleInterface` (check `Oracle.py`). In particular, it needs to implement the `ask` method. This will be used in `env.py`, lines 172-177.

You can use a remote or local VLM for this. We provide two classes (in `utils.py`): `GeminiLLM`, which calls the Gemini API, and `ClientBasedLLM`, which uses a local VLM spawned from `vllm` (see next section). 

To use GeminiLLM, you need a Gemini API Key. After obtaining the key, create a file named `.env.ml` *IN YOUR HOME DIRECTORY* and add the following line
> GEMINI_API_KEY="your api key here".

Any GeminiLLM instance will pull the key from that file. Be careful that the API usage can costs quite a bit of money, especially when using stronger models like `gemini-3-pro`. We suggest using the smaller `flash` models, which are good enough to act a fair and useful oracles. If you prefer running a local VLM, we suggest at least a 32B model.

# Questioner

Your **goal is to implement a questioner**, i.e. a multimodal model able to ask questions to the oracle or coming to a conclusion.
More precisely, at each turn the questioner will take an observation (an image of an object), compare it with an textual description, and determine whether the description matches the observation or not. If the model is unsure, because the description and image are ambiguous, it can ask questions to the oracle, which will answer faithfully. The goal is to maximize the number of correct conclusions while asking as few questions as possible.

You can implement the Questioner however you like, but it needs to inherit from the class `QuestionerInterface` (see file `Questioner.py`). As an example, we provide a questioner implementation, `QuestionerLocalVLM`, that can be used with any local VLM run via `vllm`. This uses the class `ClientBasedLLM` from `utils.py`. You will need to launch, locally via `vllm`, any VLM that you like; then, this class will connect to it (via the `ClientBasedLLM` attribute) and use it as oracle. You may implement any logic that you want inside the method `ask_or_conclude` (which is only partially implemented).

We also give you a generic class, `YourQuestioner`, that you can use as a template to implement the oracle as you like, if you don't want to use the previous implementations. However, remember to always inherit from `QuestionerInterface`.

# Evaluation

Run the file `eval_model.py` to evaluate your model. You need to add, lines 88-104, your oracle (or the provided ones). At line 155, you have to add your Questioner.

This script will evaluate your questioner on the training set using the provided oracle. The training set comprises trajectories with different description level (see the paper for more information). You can choose which level of description to use by passing the argument --description-type. There are six description types, 'category', 'color', 'context', 'color_context_feature', 'color_feature', 'color_context'. You can pass 'all': in this case, the script will loop over all other types.

Before running it, check the provided arguments by running 
> python eval_model.py --help
