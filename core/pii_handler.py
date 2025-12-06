"""
PII Detection and Scrubbing System for CogRepo

Provides comprehensive PII handling with:
- Pattern-based detection for common PII types
- Multiple scrubbing modes (redact, hash, mask, remove)
- Optional reversible vault for recovery
- Audit logging for compliance
- Configurable sensitivity levels

Supported PII types:
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- API keys and secrets
- IP addresses
- Custom patterns

Usage:
    from core.pii_handler import PIIHandler, PIIConfig

    handler = PIIHandler(PIIConfig(scrub_mode="redact"))
    clean_text, audit = handler.process(raw_text)

    if audit["pii_found"]:
        print(f"Scrubbed {audit['count']} PII instances")
"""

import re
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Pattern, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class PIIType(Enum):
    """Types of PII we detect and handle."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"
    AWS_KEY = "aws_key"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    IBAN = "iban"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    DATE_OF_BIRTH = "date_of_birth"
    CUSTOM = "custom"


class ScrubMode(Enum):
    """Available scrubbing modes."""
    REDACT = "redact"      # Replace with [REDACTED-TYPE]
    HASH = "hash"          # Replace with deterministic hash
    MASK = "mask"          # Partial masking (e.g., ***@example.com)
    REMOVE = "remove"      # Complete removal
    TOKENIZE = "tokenize"  # Replace with reversible token


class Severity(Enum):
    """PII severity levels for filtering."""
    LOW = "low"            # IPs, MAC addresses
    MEDIUM = "medium"      # Emails, phones
    HIGH = "high"          # SSN, credit cards
    CRITICAL = "critical"  # API keys, passwords


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: PIIType
    original: str
    start: int
    end: int
    confidence: float
    severity: Severity
    replacement: str = ""
    context: str = ""  # Surrounding text for audit

    def __hash__(self):
        return hash((self.pii_type, self.start, self.end))


@dataclass
class PIIConfig:
    """PII handling configuration."""
    enabled: bool = True

    # Scrubbing mode
    scrub_mode: ScrubMode = ScrubMode.REDACT
    redact_placeholder: str = "[REDACTED-{type}]"
    hash_prefix: str = "[HASH:"
    hash_suffix: str = "]"
    hash_length: int = 12

    # Detection toggles
    detect_emails: bool = True
    detect_phones: bool = True
    detect_ssn: bool = True
    detect_credit_cards: bool = True
    detect_api_keys: bool = True
    detect_aws_keys: bool = True
    detect_ip_addresses: bool = True
    detect_mac_addresses: bool = False
    detect_iban: bool = False
    detect_dob: bool = False

    # Severity filtering
    min_severity: Severity = Severity.LOW

    # Context capture (for audit)
    capture_context: bool = True
    context_chars: int = 20

    # Vault settings (for reversible scrubbing)
    enable_vault: bool = False
    vault_path: Optional[Path] = None

    # Allowlists
    allowed_domains: Set[str] = field(default_factory=set)
    allowed_ips: Set[str] = field(default_factory=set)

    # Custom patterns
    custom_patterns: Dict[str, str] = field(default_factory=dict)


@dataclass
class PIIAudit:
    """Audit record for PII processing."""
    timestamp: datetime
    text_length: int
    pii_found: bool
    total_matches: int
    matches_by_type: Dict[str, int]
    matches_by_severity: Dict[str, int]
    scrub_mode: str
    vault_entries: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "text_length": self.text_length,
            "pii_found": self.pii_found,
            "total_matches": self.total_matches,
            "matches_by_type": self.matches_by_type,
            "matches_by_severity": self.matches_by_severity,
            "scrub_mode": self.scrub_mode,
            "vault_entries": self.vault_entries
        }


# =============================================================================
# Pattern Definitions
# =============================================================================

class PIIPatterns:
    """
    Compiled regex patterns for PII detection.

    Patterns are designed to balance precision and recall.
    False positives are preferred over false negatives for security.
    """

    # Email addresses (RFC 5322 simplified)
    EMAIL = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
        re.IGNORECASE
    )

    # Phone numbers (US and international formats)
    PHONE = re.compile(
        r'(?:\+?1[-.\s]?)?'  # Optional country code
        r'(?:\(?\d{3}\)?[-.\s]?)?'  # Optional area code
        r'\d{3}[-.\s]?\d{4}\b'  # Main number
    )

    # Social Security Numbers
    SSN = re.compile(
        r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
    )

    # Credit card numbers (major brands)
    CREDIT_CARD = re.compile(
        r'\b(?:'
        r'4[0-9]{12}(?:[0-9]{3})?|'  # Visa
        r'5[1-5][0-9]{14}|'  # MasterCard
        r'3[47][0-9]{13}|'  # American Express
        r'6(?:011|5[0-9]{2})[0-9]{12}|'  # Discover
        r'(?:\d{4}[-\s]?){3}\d{4}'  # Generic with separators
        r')\b'
    )

    # API keys (various formats)
    API_KEY = re.compile(
        r'\b(?:'
        r'sk-[a-zA-Z0-9]{32,}|'  # OpenAI
        r'sk-ant-[a-zA-Z0-9-]{32,}|'  # Anthropic
        r'api[_-]?key["\'\s:=]+["\']?[a-zA-Z0-9_-]{20,}["\']?|'  # Generic
        r'bearer\s+[a-zA-Z0-9_-]{20,}|'  # Bearer tokens
        r'token["\'\s:=]+["\']?[a-zA-Z0-9_-]{20,}["\']?'  # Generic tokens
        r')\b',
        re.IGNORECASE
    )

    # AWS Access Keys
    AWS_KEY = re.compile(
        r'\b(?:'
        r'AKIA[0-9A-Z]{16}|'  # Access Key ID
        r'aws[_-]?(?:secret[_-]?)?(?:access[_-]?)?key["\'\s:=]+["\']?[A-Za-z0-9/+=]{40}["\']?'  # Secret Key
        r')\b',
        re.IGNORECASE
    )

    # IPv4 addresses
    IP_ADDRESS = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )

    # MAC addresses
    MAC_ADDRESS = re.compile(
        r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
    )

    # IBAN (International Bank Account Number)
    IBAN = re.compile(
        r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}(?:[A-Z0-9]?){0,16}\b'
    )

    # Date of birth patterns
    DOB = re.compile(
        r'\b(?:'
        r'(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}|'  # MM/DD/YYYY
        r'(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])|'  # YYYY-MM-DD
        r'(?:0?[1-9]|[12][0-9]|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}'   # DD/MM/YYYY
        r')\b'
    )

    # Password patterns in config/code
    PASSWORD = re.compile(
        r'(?:password|passwd|pwd)["\'\s:=]+["\']?[^\s"\']{8,}["\']?',
        re.IGNORECASE
    )


# =============================================================================
# PII Detector
# =============================================================================

class PIIDetector:
    """
    Detects PII in text using pattern matching.

    Thread-safe and reusable.
    """

    # Mapping of PII types to patterns and severity
    PATTERN_MAP: Dict[PIIType, Tuple[Pattern, Severity]] = {
        PIIType.EMAIL: (PIIPatterns.EMAIL, Severity.MEDIUM),
        PIIType.PHONE: (PIIPatterns.PHONE, Severity.MEDIUM),
        PIIType.SSN: (PIIPatterns.SSN, Severity.HIGH),
        PIIType.CREDIT_CARD: (PIIPatterns.CREDIT_CARD, Severity.HIGH),
        PIIType.API_KEY: (PIIPatterns.API_KEY, Severity.CRITICAL),
        PIIType.AWS_KEY: (PIIPatterns.AWS_KEY, Severity.CRITICAL),
        PIIType.IP_ADDRESS: (PIIPatterns.IP_ADDRESS, Severity.LOW),
        PIIType.MAC_ADDRESS: (PIIPatterns.MAC_ADDRESS, Severity.LOW),
        PIIType.IBAN: (PIIPatterns.IBAN, Severity.HIGH),
        PIIType.DATE_OF_BIRTH: (PIIPatterns.DOB, Severity.MEDIUM),
    }

    def __init__(self, config: PIIConfig):
        """Initialize detector with configuration."""
        self.config = config
        self._custom_patterns: Dict[str, Tuple[Pattern, Severity]] = {}
        self._compile_custom_patterns()

    def _compile_custom_patterns(self):
        """Compile custom regex patterns from config."""
        for name, pattern_str in self.config.custom_patterns.items():
            try:
                pattern = re.compile(pattern_str)
                self._custom_patterns[name] = (pattern, Severity.MEDIUM)
            except re.error as e:
                logger.warning(f"Invalid custom pattern '{name}': {e}")

    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of PII matches, sorted by position
        """
        if not text or not self.config.enabled:
            return []

        matches = []

        # Check built-in patterns
        for pii_type, (pattern, severity) in self.PATTERN_MAP.items():
            if not self._should_detect(pii_type):
                continue

            if not self._meets_severity(severity):
                continue

            for match in pattern.finditer(text):
                matched_text = match.group()

                # Skip if in allowlist
                if self._is_allowed(pii_type, matched_text):
                    continue

                # Validate match (reduce false positives)
                if not self._validate_match(pii_type, matched_text):
                    continue

                context = self._get_context(text, match.start(), match.end())

                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original=matched_text,
                    start=match.start(),
                    end=match.end(),
                    confidence=self._calculate_confidence(pii_type, matched_text),
                    severity=severity,
                    context=context
                ))

        # Check custom patterns
        for name, (pattern, severity) in self._custom_patterns.items():
            if not self._meets_severity(severity):
                continue

            for match in pattern.finditer(text):
                context = self._get_context(text, match.start(), match.end())
                matches.append(PIIMatch(
                    pii_type=PIIType.CUSTOM,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                    severity=severity,
                    context=context
                ))

        # Remove overlapping matches (keep higher severity/confidence)
        matches = self._deduplicate_matches(matches)

        return sorted(matches, key=lambda m: m.start)

    def _should_detect(self, pii_type: PIIType) -> bool:
        """Check if we should detect this PII type."""
        type_to_config = {
            PIIType.EMAIL: "detect_emails",
            PIIType.PHONE: "detect_phones",
            PIIType.SSN: "detect_ssn",
            PIIType.CREDIT_CARD: "detect_credit_cards",
            PIIType.API_KEY: "detect_api_keys",
            PIIType.AWS_KEY: "detect_aws_keys",
            PIIType.IP_ADDRESS: "detect_ip_addresses",
            PIIType.MAC_ADDRESS: "detect_mac_addresses",
            PIIType.IBAN: "detect_iban",
            PIIType.DATE_OF_BIRTH: "detect_dob",
        }
        attr = type_to_config.get(pii_type)
        return getattr(self.config, attr, True) if attr else True

    def _meets_severity(self, severity: Severity) -> bool:
        """Check if severity meets minimum threshold."""
        severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        min_idx = severity_order.index(self.config.min_severity)
        current_idx = severity_order.index(severity)
        return current_idx >= min_idx

    def _is_allowed(self, pii_type: PIIType, value: str) -> bool:
        """Check if value is in allowlist."""
        if pii_type == PIIType.EMAIL:
            domain = value.split("@")[-1].lower()
            return domain in self.config.allowed_domains
        elif pii_type == PIIType.IP_ADDRESS:
            return value in self.config.allowed_ips
        return False

    def _validate_match(self, pii_type: PIIType, value: str) -> bool:
        """
        Additional validation to reduce false positives.

        Returns True if the match appears to be valid PII.
        """
        if pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm check
            return self._luhn_check(value)
        elif pii_type == PIIType.PHONE:
            # Must have enough digits
            digits = re.sub(r'\D', '', value)
            return 7 <= len(digits) <= 15
        elif pii_type == PIIType.IP_ADDRESS:
            # Skip obviously internal/example IPs
            if value.startswith(("127.", "0.", "255.", "192.168.", "10.")):
                return False
            return True
        return True

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = [int(d) for d in re.sub(r'\D', '', card_number)]
        if len(digits) < 13:
            return False

        checksum = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        return checksum % 10 == 0

    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """
        Calculate confidence score for a match.

        Higher scores indicate more certain PII detection.
        """
        base_confidence = {
            PIIType.EMAIL: 0.95,
            PIIType.PHONE: 0.85,
            PIIType.SSN: 0.90,
            PIIType.CREDIT_CARD: 0.92,
            PIIType.API_KEY: 0.98,
            PIIType.AWS_KEY: 0.99,
            PIIType.IP_ADDRESS: 0.80,
            PIIType.MAC_ADDRESS: 0.90,
            PIIType.IBAN: 0.85,
            PIIType.DATE_OF_BIRTH: 0.70,
        }
        return base_confidence.get(pii_type, 0.75)

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get surrounding context for audit logging."""
        if not self.config.capture_context:
            return ""

        ctx_start = max(0, start - self.config.context_chars)
        ctx_end = min(len(text), end + self.config.context_chars)

        before = text[ctx_start:start]
        after = text[end:ctx_end]

        return f"...{before}[PII]{after}..."

    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping best ones."""
        if not matches:
            return []

        # Sort by severity (desc), then confidence (desc)
        severity_order = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1
        }

        sorted_matches = sorted(
            matches,
            key=lambda m: (severity_order[m.severity], m.confidence),
            reverse=True
        )

        result = []
        used_positions: Set[Tuple[int, int]] = set()

        for match in sorted_matches:
            # Check for overlap with existing matches
            overlaps = False
            for used_start, used_end in used_positions:
                if (match.start < used_end and match.end > used_start):
                    overlaps = True
                    break

            if not overlaps:
                result.append(match)
                used_positions.add((match.start, match.end))

        return result


# =============================================================================
# PII Scrubber
# =============================================================================

class PIIScrubber:
    """
    Scrubs detected PII from text.

    Supports multiple scrubbing modes:
    - redact: Replace with placeholder
    - hash: Replace with deterministic hash
    - mask: Partial masking
    - remove: Complete removal
    - tokenize: Reversible tokenization
    """

    def __init__(self, config: PIIConfig, vault: Optional['PIIVault'] = None):
        """Initialize scrubber with configuration."""
        self.config = config
        self.vault = vault

    def scrub(self, text: str, matches: List[PIIMatch]) -> Tuple[str, PIIAudit]:
        """
        Scrub PII from text.

        Args:
            text: Original text
            matches: Detected PII matches

        Returns:
            Tuple of (scrubbed_text, audit_record)
        """
        # Initialize audit
        matches_by_type: Dict[str, int] = {}
        matches_by_severity: Dict[str, int] = {}

        if not matches:
            return text, PIIAudit(
                timestamp=datetime.now(),
                text_length=len(text),
                pii_found=False,
                total_matches=0,
                matches_by_type={},
                matches_by_severity={},
                scrub_mode=self.config.scrub_mode.value
            )

        # Process matches in reverse order to preserve positions
        result = text
        vault_entries = 0

        for match in reversed(sorted(matches, key=lambda m: m.start)):
            # Get replacement
            replacement = self._get_replacement(match)
            match.replacement = replacement

            # Apply replacement
            result = result[:match.start] + replacement + result[match.end:]

            # Track statistics
            type_name = match.pii_type.value
            matches_by_type[type_name] = matches_by_type.get(type_name, 0) + 1

            sev_name = match.severity.value
            matches_by_severity[sev_name] = matches_by_severity.get(sev_name, 0) + 1

            # Store in vault if enabled
            if self.vault and self.config.scrub_mode == ScrubMode.TOKENIZE:
                self.vault.store(match)
                vault_entries += 1

        audit = PIIAudit(
            timestamp=datetime.now(),
            text_length=len(text),
            pii_found=True,
            total_matches=len(matches),
            matches_by_type=matches_by_type,
            matches_by_severity=matches_by_severity,
            scrub_mode=self.config.scrub_mode.value,
            vault_entries=vault_entries
        )

        return result, audit

    def _get_replacement(self, match: PIIMatch) -> str:
        """Get replacement text based on scrub mode."""
        mode = self.config.scrub_mode

        if mode == ScrubMode.REDACT:
            return self.config.redact_placeholder.format(
                type=match.pii_type.value.upper()
            )

        elif mode == ScrubMode.HASH:
            hash_val = hashlib.sha256(match.original.encode()).hexdigest()
            truncated = hash_val[:self.config.hash_length]
            return f"{self.config.hash_prefix}{truncated}{self.config.hash_suffix}"

        elif mode == ScrubMode.MASK:
            return self._mask_value(match)

        elif mode == ScrubMode.REMOVE:
            return ""

        elif mode == ScrubMode.TOKENIZE:
            # Generate reversible token
            token_id = hashlib.md5(
                f"{match.original}:{match.pii_type.value}".encode()
            ).hexdigest()[:8]
            return f"[TOKEN:{match.pii_type.value.upper()}:{token_id}]"

        return "[REDACTED]"

    def _mask_value(self, match: PIIMatch) -> str:
        """Partially mask a value."""
        orig = match.original

        if match.pii_type == PIIType.EMAIL:
            parts = orig.split("@")
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                if len(local) > 2:
                    masked_local = local[:2] + "*" * (len(local) - 2)
                else:
                    masked_local = "*" * len(local)
                return f"{masked_local}@{domain}"

        elif match.pii_type == PIIType.PHONE:
            digits = re.sub(r'\D', '', orig)
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]

        elif match.pii_type == PIIType.CREDIT_CARD:
            digits = re.sub(r'\D', '', orig)
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]

        elif match.pii_type == PIIType.SSN:
            return "***-**-" + orig[-4:]

        elif match.pii_type == PIIType.IP_ADDRESS:
            parts = orig.split(".")
            return f"{parts[0]}.***.***.*"

        # Default masking
        if len(orig) > 4:
            return orig[:2] + "*" * (len(orig) - 4) + orig[-2:]
        return "*" * len(orig)


# =============================================================================
# PII Vault (Reversible Tokenization)
# =============================================================================

class PIIVault:
    """
    Secure storage for reversible PII tokenization.

    Allows recovering original values when needed
    (e.g., for authorized access or debugging).
    """

    def __init__(self, config: PIIConfig):
        """Initialize vault with configuration."""
        self.config = config
        self._lock = threading.Lock()
        self._storage: Dict[str, Dict] = {}
        self._vault_path = config.vault_path

        if self._vault_path and self._vault_path.exists():
            self._load()

    def store(self, match: PIIMatch) -> str:
        """
        Store PII match and return token.

        Args:
            match: The PII match to store

        Returns:
            Token that can be used to retrieve the value
        """
        token_id = hashlib.md5(
            f"{match.original}:{match.pii_type.value}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        with self._lock:
            self._storage[token_id] = {
                "original": match.original,
                "pii_type": match.pii_type.value,
                "severity": match.severity.value,
                "stored_at": datetime.now().isoformat(),
                "context": match.context
            }

            if self._vault_path:
                self._persist()

        return token_id

    def retrieve(self, token_id: str) -> Optional[str]:
        """
        Retrieve original value from token.

        Args:
            token_id: Token returned from store()

        Returns:
            Original PII value or None if not found
        """
        with self._lock:
            entry = self._storage.get(token_id)
            return entry["original"] if entry else None

    def delete(self, token_id: str) -> bool:
        """
        Delete a stored PII value.

        Args:
            token_id: Token to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if token_id in self._storage:
                del self._storage[token_id]
                if self._vault_path:
                    self._persist()
                return True
            return False

    def clear(self) -> int:
        """
        Clear all stored PII.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._storage)
            self._storage.clear()
            if self._vault_path:
                self._persist()
            return count

    def _persist(self):
        """Persist vault to disk."""
        if not self._vault_path:
            return

        try:
            with open(self._vault_path, 'w') as f:
                json.dump(self._storage, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist PII vault: {e}")

    def _load(self):
        """Load vault from disk."""
        if not self._vault_path or not self._vault_path.exists():
            return

        try:
            with open(self._vault_path, 'r') as f:
                self._storage = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load PII vault: {e}")
            self._storage = {}


# =============================================================================
# Main PII Handler
# =============================================================================

class PIIHandler:
    """
    Main interface for PII detection and scrubbing.

    Coordinates detector, scrubber, and vault components.

    Usage:
        handler = PIIHandler(PIIConfig())
        clean_text, audit = handler.process(raw_text)
    """

    def __init__(self, config: Optional[PIIConfig] = None):
        """
        Initialize PII handler.

        Args:
            config: PII configuration (uses defaults if None)
        """
        self.config = config or PIIConfig()
        self.detector = PIIDetector(self.config)

        # Initialize vault if enabled
        self.vault = None
        if self.config.enable_vault:
            self.vault = PIIVault(self.config)

        self.scrubber = PIIScrubber(self.config, self.vault)

        # Statistics
        self._total_processed = 0
        self._total_pii_found = 0
        self._pii_by_type: Dict[str, int] = {}

    def process(self, text: str) -> Tuple[str, PIIAudit]:
        """
        Detect and scrub PII from text.

        Args:
            text: Text to process

        Returns:
            Tuple of (clean_text, audit_record)
        """
        if not self.config.enabled or not text:
            return text, PIIAudit(
                timestamp=datetime.now(),
                text_length=len(text) if text else 0,
                pii_found=False,
                total_matches=0,
                matches_by_type={},
                matches_by_severity={},
                scrub_mode=self.config.scrub_mode.value
            )

        # Detect PII
        matches = self.detector.detect(text)

        # Scrub PII
        clean_text, audit = self.scrubber.scrub(text, matches)

        # Update statistics
        self._total_processed += 1
        if audit.pii_found:
            self._total_pii_found += audit.total_matches
            for pii_type, count in audit.matches_by_type.items():
                self._pii_by_type[pii_type] = self._pii_by_type.get(pii_type, 0) + count

        return clean_text, audit

    def has_pii(self, text: str) -> bool:
        """
        Quick check if text contains PII.

        Args:
            text: Text to check

        Returns:
            True if PII detected
        """
        if not self.config.enabled or not text:
            return False
        return len(self.detector.detect(text)) > 0

    def get_pii_types(self, text: str) -> Set[PIIType]:
        """
        Get types of PII found in text.

        Args:
            text: Text to check

        Returns:
            Set of PII types found
        """
        if not self.config.enabled or not text:
            return set()

        matches = self.detector.detect(text)
        return {m.pii_type for m in matches}

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self._total_processed,
            "total_pii_found": self._total_pii_found,
            "pii_by_type": self._pii_by_type.copy(),
            "config": {
                "enabled": self.config.enabled,
                "scrub_mode": self.config.scrub_mode.value,
                "vault_enabled": self.config.enable_vault
            }
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self._total_processed = 0
        self._total_pii_found = 0
        self._pii_by_type.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def scrub_pii(
    text: str,
    mode: ScrubMode = ScrubMode.REDACT,
    **config_overrides
) -> str:
    """
    Quick PII scrubbing for simple use cases.

    Args:
        text: Text to scrub
        mode: Scrubbing mode
        **config_overrides: Additional config options

    Returns:
        Scrubbed text
    """
    config = PIIConfig(scrub_mode=mode, **config_overrides)
    handler = PIIHandler(config)
    clean_text, _ = handler.process(text)
    return clean_text


def detect_pii(text: str, **config_overrides) -> List[PIIMatch]:
    """
    Quick PII detection for simple use cases.

    Args:
        text: Text to scan
        **config_overrides: Additional config options

    Returns:
        List of PII matches
    """
    config = PIIConfig(**config_overrides)
    detector = PIIDetector(config)
    return detector.detect(text)


def has_pii(text: str) -> bool:
    """
    Quick check if text contains any PII.

    Args:
        text: Text to check

    Returns:
        True if PII detected
    """
    handler = PIIHandler()
    return handler.has_pii(text)
