import os
import sys
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
RETRIEVAL_THRESHOLD = 0.30

# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "intent_classifier.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "apple_support_labeled.csv"
)


# ================================================================
# LOAD INTENT MODEL
# ================================================================

print("=" * 80)
print("HIVER SUPPORT AGENT")
print("=" * 80)

print("\nLoading intent model:")
print(MODEL_PATH)

try:
    intent_model = joblib.load(MODEL_PATH)
    print("Intent model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load intent model: {e}")
    sys.exit(1)


# ================================================================
# LOAD SUPPORT DATA
# ================================================================

print("\nLoading support responses:")
print(DATA_PATH)

try:
    support_df = pd.read_csv(DATA_PATH)

    required_columns = [
        "customer_text",
        "brand_text",
        "intent"
    ]

    for column in required_columns:
        if column not in support_df.columns:
            raise ValueError(
                f"Required column '{column}' not found in dataset."
            )

    support_df["customer_text"] = (
        support_df["customer_text"]
        .fillna("")
        .astype(str)
    )

    support_df["brand_text"] = (
        support_df["brand_text"]
        .fillna("")
        .astype(str)
    )

    support_df["intent"] = (
        support_df["intent"]
        .fillna("")
        .astype(str)
    )

    support_df = support_df[
        (support_df["customer_text"].str.strip() != "") &
        (support_df["brand_text"].str.strip() != "")
    ].copy()

    print(f"Support examples loaded: {len(support_df)}")

except Exception as e:
    print(f"ERROR: Could not load support data: {e}")
    sys.exit(1)


# ================================================================
# CLEAN TEXT FOR SEARCH
# ================================================================

def clean_text(text):
    """
    Clean text before TF-IDF similarity search.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter/X mentions
    text = re.sub(r"@\w+", " ", text)

    # Replace HTML entities
    text = text.replace("&gt;", " ")
    text = text.replace("&lt;", " ")
    text = text.replace("&amp;", "and")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


support_df["search_text"] = (
    support_df["customer_text"]
    .apply(clean_text)
)


# ================================================================
# BUILD TF-IDF SEARCH INDEX
# ================================================================

print("\nBuilding response retrieval index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

customer_vectors = vectorizer.fit_transform(
    support_df["search_text"]
)

print("Response retrieval index ready.")


# ================================================================
# INTENT CLASSIFICATION
# ================================================================

def classify_intent(message):

    if not message or not message.strip():
        return "unknown"

    return intent_model.predict([message])[0]


# ================================================================
# CLEAN SUPPORT RESPONSE
# ================================================================

def clean_response(response):
    """
    Remove Twitter/X-specific artifacts from retrieved responses.
    """

    response = str(response).strip()

    # Remove URLs
    response = re.sub(r"https?://\S+", "", response)

    # Remove @mentions
    response = re.sub(r"@\w+", "", response)

    # Remove excessive whitespace
    response = re.sub(r"\s+", " ", response).strip()

    return response


# ================================================================
# CHECK RESPONSE QUALITY
# ================================================================

def is_good_response(response):

    if not response:
        return False

    response_lower = response.lower()

    # Reject obvious social-media promotional responses
    bad_patterns = [
        "introducing the new control center",
        "learn more",
        "check out",
        "follow us",
        "giveaway",
        "contest",
        "promo",
        "promotion",
        "visit our website"
    ]

    for pattern in bad_patterns:
        if pattern in response_lower:
            return False

    # Reject responses that are too short
    if len(response.split()) < 5:
        return False

    return True


# ================================================================
# RESPONSE RETRIEVAL
# ================================================================

def retrieve_response(message, intent):

    # Only search examples belonging to the detected intent
    candidates = support_df[
        support_df["intent"] == intent
    ].copy()

    if candidates.empty:
        return None

    # Clean the customer's message
    cleaned_message = clean_text(message)

    # Transform customer message
    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    # Transform only candidate examples
    candidate_vectors = vectorizer.transform(
        candidates["search_text"]
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        message_vector,
        candidate_vectors
    )[0]

    # Get best matching example
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    best_response = candidates.iloc[
        best_index
    ]["brand_text"]

    best_response = clean_response(best_response)

    print(f"[Retrieval similarity: {best_score:.3f}]")

    # Use retrieval only for strong matches
    if best_score < RETRIEVAL_THRESHOLD:
    return None

    # Reject poor-quality responses
    if not is_good_response(best_response):
        return None

    return best_response


# ================================================================
# FALLBACK RESPONSES
# ================================================================

def fallback_response(intent):

    responses = {

        "battery_issue":
            "I understand that your iPhone battery is draining quickly. "
            "Please check Settings > Battery to see which apps are using "
            "the most battery. You can also enable Low Power Mode.",

        "ios_update_issue":
            "It looks like you're having a problem with an iOS update. "
            "Please make sure your iPhone has enough storage, is connected "
            "to Wi-Fi, and has sufficient battery before trying again.",

        "app_issue":
            "It looks like you're having a problem with an app. "
            "Please tell me which app is affected and whether it crashes, "
            "freezes, or refuses to open.",

        "iphone_freezing":
            "It sounds like your iPhone is freezing or becoming "
            "unresponsive. Try restarting the iPhone first. "
            "If the problem continues, tell me when the freezing occurs.",

        "itunes_issue":
            "It looks like you're having an issue with iTunes. "
            "Please tell me whether you're having trouble connecting, "
            "syncing, backing up, or restoring your iPhone.",

        "apple_id_issue":
            "It sounds like you're having an Apple ID issue. "
            "Please tell me whether you cannot sign in, forgot your "
            "password, or are seeing an error message.",

        "wifi_connectivity":
            "It looks like you're experiencing a Wi-Fi or connectivity "
            "issue. Try turning Wi-Fi off and back on, then reconnect "
            "to the network.",

        "control_center":
            "It sounds like you're having an issue with Control Center. "
            "Please tell me which control is not working correctly.",

        "apple_music":
            "It looks like you're having an Apple Music issue. "
            "Please tell me whether the problem involves playback, "
            "your library, playlists, syncing, or your subscription.",

        "touch_screen":
            "It sounds like your touchscreen or Touch ID isn't responding "
            "correctly. Please tell me whether the entire screen or only "
            "a specific area is affected.",

        "facetime":
            "It looks like you're having a FaceTime issue. "
            "Please tell me whether you cannot make calls, receive calls, "
            "or have an audio or video problem.",

        "iphone_device_issue":
            "I understand you're having an issue with your iPhone. "
            "Could you describe exactly what is happening with the device?",

        "ipad_issue":
            "It looks like you're having an issue with your iPad. "
            "Please describe what is happening and I'll help troubleshoot it.",

        "feedback":
            "Thanks for sharing your feedback. "
            "Please tell me what you'd like Apple to improve or change.",

        "support_followup":
            "I understand you're following up with support. "
            "Please tell me what issue you're waiting for help with."
    }

    return responses.get(
        intent,
        "I understand you're experiencing an issue. "
        "Could you tell me a little more about the problem?"
    )


# ================================================================
# RESPONSE GENERATION
# ================================================================

def generate_response(message, intent):

    retrieved = retrieve_response(
        message,
        intent
    )

    if retrieved:
        return retrieved

    return fallback_response(intent)


# ================================================================
# COMPLETE AGENT
# ================================================================

def run_agent(message):

    intent = classify_intent(message)

    response = generate_response(
        message,
        intent
    )

    return {
        "message": message,
        "intent": intent,
        "response": response
    }


# ================================================================
# TEST MODE
# ================================================================

if __name__ == "__main__":

    print("\nType a customer message to test the support agent.")
    print("Type 'exit' to quit.\n")

    while True:

        try:
            message = input("Customer: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent.")
            break

        if message.lower() == "exit":
            print("\nExiting agent.")
            break

        if not message:
            continue

        result = run_agent(message)

        print(f"\nDetected intent: {result['intent']}")
        print(f"Agent: {result['response']}\n")
