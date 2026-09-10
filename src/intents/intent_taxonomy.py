"""
Curated intent taxonomy for the Hiver Support Agent.

The taxonomy is derived from the AppleSupport dataset and the
intent-discovery clustering results.

Each intent has:
- intent_id
- name
- description
- keywords
"""

INTENTS = {
    "battery_issue": {
        "name": "Battery / Battery Drain",
        "description": "Problems involving battery draining, battery life, or charging.",
        "keywords": [
            "battery",
            "battery drain",
            "battery draining",
            "battery life",
            "charge",
            "charging",
            "not charging",
        ],
    },

    "ios_update_issue": {
        "name": "iOS Update Issue",
        "description": "Problems caused by or related to an iOS update.",
        "keywords": [
            "ios update",
            "software update",
            "ios 11",
            "update",
            "updated",
            "latest update",
            "new update",
            "after update",
        ],
    },

    "app_issue": {
        "name": "App Problem",
        "description": "An Apple or third-party app is not opening, responding, or behaving correctly.",
        "keywords": [
            "app",
            "apps",
            "application",
            "app not working",
            "app crashes",
            "app won't open",
            "app freezing",
        ],
    },

    "iphone_freezing": {
        "name": "iPhone Freezing / Performance",
        "description": "Device freezing, lagging, slowing down, restarting, or becoming unresponsive.",
        "keywords": [
            "freezing",
            "freeze",
            "freezes",
            "frozen",
            "lag",
            "lagging",
            "slow",
            "unresponsive",
            "restart",
            "restarting",
        ],
    },

    "itunes_issue": {
        "name": "iTunes Problem",
        "description": "Problems involving iTunes, syncing, restoring, or connecting an iPhone to iTunes.",
        "keywords": [
            "itunes",
            "iTunes",
            "sync",
            "itunes error",
            "itunes restore",
            "itunes backup",
            "itunes connect",
        ],
    },

    "apple_id_issue": {
        "name": "Apple ID / Account",
        "description": "Issues involving Apple ID, account access, authentication, or account-related services.",
        "keywords": [
            "apple id",
            "account",
            "login",
            "sign in",
            "password",
            "authentication",
            "icloud account",
        ],
    },

    "wifi_connectivity": {
        "name": "Wi-Fi / Connectivity",
        "description": "Problems connecting to or maintaining Wi-Fi and other network connections.",
        "keywords": [
            "wifi",
            "wi-fi",
            "internet",
            "network",
            "connection",
            "connected",
            "disconnect",
            "bluetooth",
        ],
    },

    "control_center": {
        "name": "Control Center",
        "description": "Problems with Control Center controls, settings, or functionality.",
        "keywords": [
            "control center",
            "control centre",
            "swipe up",
            "control panel",
            "wifi control",
            "bluetooth control",
            "music control",
        ],
    },

    "apple_music": {
        "name": "Apple Music",
        "description": "Problems involving Apple Music, music playback, library, syncing, or subscriptions.",
        "keywords": [
            "apple music",
            "music",
            "songs",
            "song",
            "music library",
            "playlist",
            "music app",
            "sync music",
            "headphones",
        ],
    },

    "touch_screen": {
        "name": "Touch Screen / Touch ID",
        "description": "Problems involving touchscreen input, touch responsiveness, or Touch ID.",
        "keywords": [
            "touch",
            "touch screen",
            "touchscreen",
            "screen",
            "touch id",
            "fingerprint",
            "not responding",
        ],
    },

    "facetime": {
        "name": "FaceTime",
        "description": "Problems with FaceTime calls, ringing, video, or receiving calls.",
        "keywords": [
            "facetime",
            "face time",
            "video call",
            "facetime call",
            "facetime not ringing",
        ],
    },

    "iphone_device_issue": {
        "name": "General iPhone Device Issue",
        "description": "General iPhone hardware or software problems that do not fit a more specific intent.",
        "keywords": [
            "iphone",
            "phone",
            "device",
            "iphone issue",
            "iphone problem",
            "phone problem",
        ],
    },

    "ipad_issue": {
        "name": "iPad Problem",
        "description": "General problems involving an iPad.",
        "keywords": [
            "ipad",
            "ipad issue",
            "ipad problem",
            "ipad not working",
        ],
    },

    "feedback": {
        "name": "Feedback",
        "description": "Customer feedback, suggestions, complaints, or comments about Apple's products or support.",
        "keywords": [
            "feedback",
            "suggestion",
            "suggest",
            "feature request",
            "comment",
        ],
    },

    "support_followup": {
        "name": "Support Follow-up / Waiting for Response",
        "description": "Customer is waiting for a response, following up with support, or communicating through DM.",
        "keywords": [
            "waiting",
            "waiting for response",
            "response",
            "reply",
            "dm",
            "direct message",
            "message",
            "sent you a dm",
        ],
    },
}


# ---------------------------------------------------------
# Intent IDs
# ---------------------------------------------------------

INTENT_IDS = list(INTENTS.keys())


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_intent(intent_id):
    """Return the definition for an intent."""
    return INTENTS.get(intent_id)


def get_all_intents():
    """Return the complete intent taxonomy."""
    return INTENTS


def get_intent_names():
    """Return intent IDs and human-readable names."""
    return {
        intent_id: intent["name"]
        for intent_id, intent in INTENTS.items()
    }


if __name__ == "__main__":
    print("=" * 80)
    print("HIVER SUPPORT AGENT — INTENT TAXONOMY")
    print("=" * 80)

    print(f"\nTotal intents: {len(INTENTS)}\n")

    for intent_id, intent in INTENTS.items():
        print(f"{intent_id}")
        print(f"  Name: {intent['name']}")
        print(f"  Description: {intent['description']}")
        print(f"  Keywords: {', '.join(intent['keywords'])}")
        print()