"""Teacher-authored topic definitions used to seed the database.

Each topic bounds what the NLP component evaluates: required concepts (with
descriptions the embeddings compare against), known misconceptions (the wrong
claim + the correct contrast), probe questions for the dialogue engine, and
one activity per learning state for the recommender.
"""

TOPICS = [
    {
        "name": "Backpropagation",
        "description": "How neural networks learn: computing gradients of the loss and updating weights.",
        "reference_explanation": (
            "A neural network makes a prediction and a loss function measures how wrong the prediction is. "
            "Backpropagation applies the chain rule to compute the gradient of the loss with respect to every "
            "weight, propagating the error signal backwards from the output layer through the hidden layers. "
            "An optimizer such as gradient descent then uses these gradients to update the weights in small "
            "steps so that the loss decreases over many iterations."
        ),
        "opening_prompt": "Teach me what you understand about backpropagation, as if I have never learned it.",
        "extension_question": "Nicely covered. What is the difference between backpropagation and gradient descent, and why do we need both?",
        "concepts": [
            {
                "name": "Loss / error",
                "description": "A loss or error function measures how far the network's prediction is from the true target value.",
                "main_question": "When a network makes a prediction, how do we know how wrong it was?",
                "easier_question": "Is there something that measures the size of the network's mistake? What is it usually called?",
                "probe_question": "How does the network know it made a mistake? What does the loss function measure?",
                "application_question": "If the loss suddenly dropped to almost zero, what would that tell you about the predictions?",
            },
            {
                "name": "Gradient",
                "description": "The gradient is the derivative of the loss with respect to each weight, showing how a small change in the weight changes the loss.",
                "main_question": "What does a gradient tell us?",
                "easier_question": "Does the gradient tell us how the loss changes when we nudge a weight?",
                "probe_question": "Can you explain what role gradients play in backpropagation?",
                "application_question": "What would a very large gradient for one particular weight suggest?",
            },
            {
                "name": "Backward propagation of error",
                "description": "The error signal is propagated backwards layer by layer using the chain rule of calculus.",
                "main_question": "Why is it called BACK-propagation — what actually moves backwards?",
                "easier_question": "Does the error signal start at the output and travel back towards the input?",
                "probe_question": "Which rule from calculus lets the error be passed backwards layer by layer?",
                "application_question": "In a ten-layer network, would the first layer still receive an error signal? How?",
            },
            {
                "name": "Weight update",
                "description": "Weights are adjusted in the opposite direction of the gradient, in small steps controlled by the learning rate.",
                "main_question": "Once we have the gradients, what happens to the weights?",
                "easier_question": "Do the weights move in the direction that increases the loss, or decreases it?",
                "probe_question": "Once the gradients are known, how do the weights actually get changed?",
                "application_question": "What could go wrong if every weight update step were very large?",
            },
            {
                "name": "Optimization / iteration",
                "description": "Training repeats forward pass, loss computation, backward pass and weight update many times to gradually minimise the loss.",
                "main_question": "Does this whole process happen once, or many times?",
                "easier_question": "Is the network trained in one single step, or gradually over many rounds?",
                "probe_question": "Does this happen once or many times? How does the network improve over training?",
                "application_question": "How would you decide when to stop training?",
            },
        ],
        "relationships": [
            {
                "source": "Backpropagation", "label": "computes", "target": "Gradient",
                "description": "Backpropagation computes the gradient of the loss for every weight.",
                "probe_question": "What does backpropagation actually calculate?",
            },
            {
                "source": "Gradient", "label": "describes change of loss w.r.t.", "target": "Weight",
                "description": "The gradient describes how the loss changes with respect to each weight.",
                "probe_question": "What does the gradient tell us about a weight?",
            },
            {
                "source": "Gradient descent", "label": "uses", "target": "Gradient",
                "description": "Gradient descent uses the gradients to decide how to change the weights.",
                "probe_question": "What does gradient descent do with the gradients?",
            },
            {
                "source": "Gradient descent", "label": "updates", "target": "Weight",
                "description": "Gradient descent updates the weights in the direction that reduces the loss.",
                "contradiction": "Gradient descent updates the weights in the direction that increases the loss.",
                "probe_question": "In which direction does gradient descent move the weights?",
            },
            {
                "source": "Weight update", "label": "aims to reduce", "target": "Loss",
                "description": "The weights are updated so that the loss decreases over time.",
                "contradiction": "The weights are updated so that the loss increases over time.",
                "probe_question": "Why do we change the weights at all — what should happen to the loss?",
            },
        ],
        "misconceptions": [
            {
                "name": "Backpropagation is the same as gradient descent",
                "description": "Backpropagation and gradient descent are the same thing, just one algorithm with two names.",
                "clarification": "Backpropagation computes the gradients; gradient descent is a separate optimizer that uses those gradients to update the weights.",
                "probe_question": "You suggested backpropagation and gradient descent are the same. What does each one actually do?",
            },
            {
                "name": "Gradients directly change the weights",
                "description": "The gradient itself directly changes the weight values without any optimizer or learning rate.",
                "clarification": "Gradients only indicate direction and sensitivity; an optimizer scales them with a learning rate to update the weights.",
                "probe_question": "You said gradients change the weights by themselves. What controls how big each weight change is?",
            },
            {
                "name": "Backpropagation changes the input",
                "description": "Backpropagation works by changing the input data so that the network gets the right answer.",
                "clarification": "Backpropagation never modifies the input data; it only computes gradients so the network's weights can be updated.",
                "probe_question": "You mentioned that backpropagation changes the input. Why would the input need to change?",
            },
            {
                "name": "Error is fixed by editing the prediction",
                "description": "The network reduces error by directly editing or correcting its output prediction after seeing the answer.",
                "clarification": "The prediction is not edited after the fact; the weights are updated so that future predictions are better.",
                "probe_question": "You implied the network corrects its prediction directly. What actually changes inside the network?",
            },
        ],
        "activities": [
            {"target_state": "not_trying", "kind": "re_engagement", "title": "One-line warm-up",
             "description": "Answer in one sentence: what is a neural network trying to minimise during training?"},
            {"target_state": "unclear", "kind": "concept_review", "title": "Hiker-on-a-hill analogy",
             "description": "Read the 'hiker descending a foggy hill' analogy for gradients, then answer: what does the slope of the hill represent?"},
            {"target_state": "struggling", "kind": "guided_practice", "title": "Guided chain-rule walkthrough",
             "description": "Step through backpropagation on a 2-layer network with 1 weight per layer, computing each derivative with hints."},
            {"target_state": "understanding", "kind": "application", "title": "Learning-rate experiment",
             "description": "Predict what happens to training when the learning rate is 10x too large, then check your prediction against provided loss curves."},
            {"target_state": "confident", "kind": "challenge", "title": "Vanishing gradients edge case",
             "description": "Explain why gradients can vanish in deep sigmoid networks and propose two mitigations. Then teach backpropagation to a classmate."},
        ],
    },
    {
        "name": "Overfitting and Regularization",
        "description": "Why models memorise training data, how to detect it, and how regularization helps.",
        "reference_explanation": (
            "Overfitting happens when a model learns the training data too specifically, including its noise, "
            "so it performs well on training data but poorly on new unseen data. It is detected by comparing "
            "training performance with validation or test performance: a large gap signals overfitting. "
            "Regularization techniques such as L1 or L2 penalties, dropout, early stopping or getting more data "
            "constrain the model's complexity so that it generalises better instead of memorising."
        ),
        "opening_prompt": "Teach me about overfitting and how regularization helps, as if I have never heard of it.",
        "extension_question": "Well explained. When could a model be UNDER-fitting instead, and how would you tell the two apart?",
        "concepts": [
            {
                "name": "Overfitting",
                "description": "The model fits the training data too closely, memorising noise instead of learning the general pattern.",
                "main_question": "In your own words, what does it mean when a model overfits?",
                "easier_question": "Is overfitting closer to memorising the data, or to understanding it?",
                "probe_question": "What actually goes wrong inside a model when we say it overfits?",
                "application_question": "Why would an overfit model do badly on data it has never seen?",
            },
            {
                "name": "Generalisation gap",
                "description": "Overfitting shows up as good performance on training data but much worse performance on validation or unseen test data.",
                "main_question": "How would you detect that a model is overfitting?",
                "easier_question": "Would you compare its performance on the training data with its performance on unseen data?",
                "probe_question": "How would you detect that a trained model is overfitting? What would you compare?",
                "application_question": "Training accuracy is 99% but test accuracy is 62% — what does that tell you?",
            },
            {
                "name": "Model complexity",
                "description": "Very flexible or complex models with many parameters are more prone to overfitting than simpler models.",
                "main_question": "What kinds of models overfit more easily?",
                "easier_question": "Does a model with many parameters overfit more easily than a very simple one?",
                "probe_question": "What kinds of models overfit more easily, and why?",
                "application_question": "Why does a tiny linear model rarely overfit?",
            },
            {
                "name": "Regularization penalty",
                "description": "Regularization such as L1 or L2 adds a penalty on large weights to the loss function, discouraging overly complex solutions.",
                "main_question": "How does something like L2 regularization help against overfitting?",
                "easier_question": "Does L2 regularization add a penalty for large weights to the loss?",
                "probe_question": "How does a technique like L2 regularization actually change what the model learns?",
                "application_question": "What might happen if the regularization penalty were far too strong?",
            },
            {
                "name": "Validation-based control",
                "description": "Techniques like early stopping, dropout, cross-validation or collecting more data reduce overfitting in practice.",
                "main_question": "Apart from a penalty term, what practical tricks reduce overfitting?",
                "easier_question": "Have you heard of early stopping or dropout? What does one of them do?",
                "probe_question": "Apart from adding a penalty term, what practical techniques reduce overfitting?",
                "application_question": "You have very little training data — which technique would you reach for first, and why?",
            },
        ],
        "relationships": [
            {
                "source": "Overfitting", "label": "shows up as", "target": "Generalisation gap",
                "description": "Overfitting shows up as a large gap between training performance and validation performance.",
                "probe_question": "If a model overfits, what would you see when comparing training and validation results?",
            },
            {
                "source": "Model complexity", "label": "increases risk of", "target": "Overfitting",
                "description": "More complex models with many parameters overfit more easily.",
                "probe_question": "How does the complexity of a model affect its tendency to overfit?",
            },
            {
                "source": "Regularization", "label": "constrains", "target": "Model complexity",
                "description": "Regularization penalises large weights so the model stays simpler.",
                "contradiction": "Regularization deletes noisy training examples so the model stays simpler.",
                "probe_question": "What does regularization actually act on to keep the model simple?",
            },
            {
                "source": "Regularization", "label": "improves", "target": "Generalisation",
                "description": "Regularization reduces overfitting so the model performs better on unseen data.",
                "contradiction": "Regularization improves performance on the training data by fitting it more closely.",
                "probe_question": "Which data does regularization help the model perform better on?",
            },
        ],
        "misconceptions": [
            {
                "name": "More training always helps",
                "description": "Training a model for more epochs always makes it better on new data.",
                "clarification": "Training longer can increase overfitting; validation performance can get worse even while training loss keeps falling.",
                "probe_question": "You suggested more training always improves the model. What happens to validation error late in training?",
            },
            {
                "name": "High training accuracy means a good model",
                "description": "If the model gets a very high accuracy on the training set, it is a good model.",
                "clarification": "High training accuracy alone can mean memorisation; quality is judged on unseen validation or test data.",
                "probe_question": "You pointed at training accuracy. Which dataset actually tells you if the model is good?",
            },
            {
                "name": "Regularization removes training data",
                "description": "Regularization works by deleting or removing noisy examples from the training data.",
                "clarification": "Regularization does not remove data; it constrains the model, for example by penalising large weights or dropping units during training.",
                "probe_question": "You said regularization removes data. What does an L2 penalty actually act on?",
            },
        ],
        "activities": [
            {"target_state": "not_trying", "kind": "re_engagement", "title": "One-line warm-up",
             "description": "Answer in one sentence: what is the difference between memorising and learning?"},
            {"target_state": "unclear", "kind": "concept_review", "title": "Exam-cramming analogy",
             "description": "Read the analogy of a student memorising past papers vs understanding the subject, then map each part to model training."},
            {"target_state": "struggling", "kind": "guided_practice", "title": "Guided curve reading",
             "description": "Given three pairs of training/validation loss curves, decide with hints which show overfitting, underfitting, or a good fit."},
            {"target_state": "understanding", "kind": "application", "title": "Pick the fix",
             "description": "For four described scenarios (small data, huge model, noisy labels, long training), choose the most suitable regularization strategy and justify it."},
            {"target_state": "confident", "kind": "challenge", "title": "Double descent teaser",
             "description": "Research the 'double descent' phenomenon and explain how it complicates the classic overfitting story. Teach your summary to a classmate."},
        ],
    },
    {
        "name": "Hidden Markov Models",
        "description": "Modelling sequences with hidden states, transitions and observable emissions.",
        "reference_explanation": (
            "A hidden Markov model describes a system that moves between hidden states over time, where the "
            "states themselves cannot be observed directly. Each hidden state emits observable outputs with "
            "certain probabilities, and transitions between states follow the Markov property: the next state "
            "depends only on the current state. Given a sequence of observations, algorithms such as Viterbi "
            "can infer the most likely sequence of hidden states, and the model's transition and emission "
            "probabilities can be learned from data."
        ),
        "opening_prompt": "Teach me what a Hidden Markov Model is, as if I have never learned it.",
        "extension_question": "Good. Why is the 'hidden' part important - what real problem could you NOT solve with a plain Markov chain?",
        "concepts": [
            {
                "name": "Hidden states",
                "description": "The system is in one of several hidden states that cannot be observed directly.",
                "main_question": "What is 'hidden' in a hidden Markov model?",
                "easier_question": "Can we directly see which state the system is in at each moment?",
                "probe_question": "If the states cannot be seen, how do we ever learn anything about them?",
                "application_question": "In the weather-and-umbrellas example, what would the hidden state be?",
            },
            {
                "name": "Observations / emissions",
                "description": "Each hidden state emits observable outputs according to emission probabilities; we only see these observations.",
                "main_question": "If the states are hidden, what do we actually get to see?",
                "easier_question": "Does each hidden state produce visible outputs with certain probabilities?",
                "probe_question": "If the states are hidden, what do we actually get to see, and how is it linked to the states?",
                "application_question": "Could two different hidden states produce the same observation?",
            },
            {
                "name": "Transition probabilities",
                "description": "The model moves between hidden states over time according to transition probabilities.",
                "main_question": "How does the model describe moving between states over time?",
                "easier_question": "Are there probabilities for jumping from one state to another?",
                "probe_question": "How does the model describe change over time between the hidden states?",
                "application_question": "What would the transition matrix look like if the state never changed?",
            },
            {
                "name": "Markov property",
                "description": "The next state depends only on the current state, not on the whole earlier history.",
                "main_question": "What does the next state depend on?",
                "easier_question": "Does the next state depend only on the current state, or on the whole history?",
                "probe_question": "What assumption does the model make about how the next state depends on the past?",
                "application_question": "Can you think of a sequence where that assumption might be unrealistic?",
            },
            {
                "name": "State inference / decoding",
                "description": "Algorithms like Viterbi infer the most likely sequence of hidden states from the observation sequence.",
                "main_question": "Given only the observations, how do we figure out which hidden states the system went through?",
                "easier_question": "Is there an algorithm that finds the most likely sequence of hidden states?",
                "probe_question": "Given only the observations, how can we figure out which hidden states the system went through?",
                "application_question": "Why decode over the whole observation sequence instead of one step at a time?",
            },
        ],
        "relationships": [
            {
                "source": "Hidden state", "label": "emits", "target": "Observation",
                "description": "Each hidden state emits the visible observations with certain probabilities.",
                "probe_question": "How are the things we observe connected to the hidden states?",
            },
            {
                "source": "Transition probabilities", "label": "govern", "target": "State change",
                "description": "Transition probabilities describe how the system moves between hidden states over time.",
                "probe_question": "What controls how the hidden state changes from one step to the next?",
            },
            {
                "source": "Markov property", "label": "constrains", "target": "Transitions",
                "description": "The next state depends only on the current state, not on the whole earlier history.",
                "contradiction": "The next state depends on the entire history of all previous states.",
                "probe_question": "How much of the past matters for predicting the next state?",
            },
            {
                "source": "Viterbi", "label": "infers", "target": "Hidden states",
                "description": "The Viterbi algorithm infers the most likely sequence of hidden states from the observations.",
                "probe_question": "Given the observations, how do we recover the hidden states?",
            },
        ],
        "misconceptions": [
            {
                "name": "States are directly observable",
                "description": "In an HMM you can directly see which state the system is in at each time step.",
                "clarification": "The states are hidden by definition; only the emitted observations are visible and the states must be inferred.",
                "probe_question": "You said we can see the states. If we could, what would be left for the model to infer?",
            },
            {
                "name": "Next state depends on the whole history",
                "description": "The next hidden state depends on the entire sequence of all previous states and observations.",
                "clarification": "Under the Markov property the next state depends only on the current state, which is what keeps the model tractable.",
                "probe_question": "You suggested the whole history matters for the next state. What does the Markov assumption say?",
            },
            {
                "name": "Observations equal states",
                "description": "The observations are the states - each output tells you exactly which state produced it.",
                "clarification": "Different states can emit the same observation with different probabilities, so an observation does not uniquely identify a state.",
                "probe_question": "You treated observations and states as the same thing. Can two different states produce the same observation?",
            },
        ],
        "activities": [
            {"target_state": "not_trying", "kind": "re_engagement", "title": "One-line warm-up",
             "description": "Answer in one sentence: give one everyday example of something you infer without seeing it directly."},
            {"target_state": "unclear", "kind": "concept_review", "title": "Weather-and-umbrella analogy",
             "description": "Read the classic example of inferring weather (hidden) from whether people carry umbrellas (observed), then label states vs observations."},
            {"target_state": "struggling", "kind": "guided_practice", "title": "Guided 2-state walkthrough",
             "description": "Hand-compute one step of state inference on a 2-state HMM with given transition and emission tables, with hints."},
            {"target_state": "understanding", "kind": "application", "title": "Model a new problem",
             "description": "Design an HMM (states, observations, rough probabilities) for detecting whether a typist is tired from their keystroke timings."},
            {"target_state": "confident", "kind": "challenge", "title": "Limits of the Markov property",
             "description": "Describe a sequence problem where the Markov assumption clearly breaks, and sketch how you would work around it. Teach your example to a classmate."},
        ],
    },
]

DEMO_STUDENTS = [
    {"name": "Aarav Shah", "program": "B.Tech CSE"},
    {"name": "Diya Patel", "program": "B.Tech CSE"},
    {"name": "Kabir Mehta", "program": "B.Tech AI & DS"},
    {"name": "Ishita Rao", "program": "B.Tech AI & DS"},
    {"name": "Rohan Nair", "program": "MBA Tech"},
    {"name": "Ananya Iyer", "program": "B.Tech CSE"},
    {"name": "Vivaan Desai", "program": "B.Tech CSE"},
    {"name": "Sara Kulkarni", "program": "MBA Tech"},
    # appended last so the earlier students keep their exact seeded histories
    {"name": "Shreshtha Bindal", "program": "B.Tech CE", "roll_no": "B023"},
]
