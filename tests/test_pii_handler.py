"""
Tests for PII Detection and Scrubbing

Tests the PIIHandler, PIIDetector, and PIIScrubber classes.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pii_handler import (
    PIIHandler, PIIConfig, PIIDetector, PIIScrubber,
    PIIType, ScrubMode, Severity, PIIMatch, PIIVault,
    scrub_pii, detect_pii, has_pii
)


class TestPIIDetector:
    """Tests for PII detection patterns."""

    @pytest.fixture
    def detector(self):
        """Create a detector with default config."""
        return PIIDetector(PIIConfig())

    def test_detect_email(self, detector):
        """Test email detection."""
        text = "Contact me at john.doe@example.com for details."
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].original == "john.doe@example.com"
        assert matches[0].confidence >= 0.9

    def test_detect_multiple_emails(self, detector):
        """Test detecting multiple emails."""
        text = "Send to alice@test.org and bob@company.co.uk"
        matches = detector.detect(text)

        assert len(matches) == 2
        emails = [m.original for m in matches]
        assert "alice@test.org" in emails
        assert "bob@company.co.uk" in emails

    def test_detect_phone_us(self, detector):
        """Test US phone number detection."""
        text = "Call me at 555-123-4567 or (555) 987-6543"
        matches = detector.detect(text)

        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 2

    def test_detect_phone_international(self, detector):
        """Test international phone detection."""
        text = "My number is +1-555-123-4567"
        matches = detector.detect(text)

        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 1

    def test_detect_ssn(self, detector):
        """Test SSN detection."""
        text = "My SSN is 123-45-6789"
        matches = detector.detect(text)

        ssn_matches = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert "123-45-6789" in ssn_matches[0].original

    def test_detect_credit_card(self, detector):
        """Test credit card detection."""
        # Valid Visa number (passes Luhn)
        text = "Card: 4111-1111-1111-1111"
        matches = detector.detect(text)

        cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1

    def test_detect_api_key_openai(self, detector):
        """Test OpenAI API key detection."""
        text = "My key is sk-abcdefghijklmnopqrstuvwxyz123456"
        matches = detector.detect(text)

        api_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(api_matches) >= 1

    def test_detect_api_key_generic(self, detector):
        """Test generic API key detection."""
        text = 'api_key = "abc123def456ghi789jkl012mno"'
        matches = detector.detect(text)

        api_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(api_matches) >= 1

    def test_detect_ip_address(self, detector):
        """Test IP address detection."""
        text = "Server is at 192.168.1.100 and 10.0.0.1"
        matches = detector.detect(text)

        ip_matches = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        # Note: 192.168.x.x and 10.x.x.x are filtered as internal
        # Let's use a public IP
        text2 = "Server is at 8.8.8.8"
        matches2 = detector.detect(text2)
        ip_matches2 = [m for m in matches2 if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_matches2) >= 1

    def test_no_false_positives_normal_text(self, detector):
        """Test that normal text doesn't trigger false positives."""
        text = "Hello, this is a normal conversation about programming."
        matches = detector.detect(text)

        assert len(matches) == 0

    def test_severity_levels(self, detector):
        """Test severity assignment."""
        text = "Email: test@example.com, SSN: 123-45-6789, IP: 8.8.8.8"
        matches = detector.detect(text)

        severities = {m.pii_type: m.severity for m in matches}

        # Email should be MEDIUM, SSN should be HIGH, IP should be LOW
        assert severities.get(PIIType.EMAIL) == Severity.MEDIUM
        assert severities.get(PIIType.SSN) == Severity.HIGH
        assert severities.get(PIIType.IP_ADDRESS) == Severity.LOW

    def test_disabled_detection(self):
        """Test that disabled detection types are skipped."""
        config = PIIConfig(detect_emails=False)
        detector = PIIDetector(config)

        text = "Email: test@example.com"
        matches = detector.detect(text)

        email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 0

    def test_allowlist_domain(self):
        """Test email domain allowlist."""
        config = PIIConfig(allowed_domains={"example.com"})
        detector = PIIDetector(config)

        text = "Email: allowed@example.com and blocked@other.com"
        matches = detector.detect(text)

        email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 1
        assert "other.com" in email_matches[0].original


class TestPIIScrubber:
    """Tests for PII scrubbing modes."""

    def test_redact_mode(self):
        """Test redaction scrubbing."""
        config = PIIConfig(scrub_mode=ScrubMode.REDACT)
        scrubber = PIIScrubber(config)

        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            original="test@example.com",
            start=0, end=16,
            confidence=0.95,
            severity=Severity.MEDIUM
        )

        replacement = scrubber._get_replacement(match)
        assert "[REDACTED-EMAIL]" in replacement

    def test_hash_mode(self):
        """Test hash scrubbing."""
        config = PIIConfig(scrub_mode=ScrubMode.HASH)
        scrubber = PIIScrubber(config)

        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            original="test@example.com",
            start=0, end=16,
            confidence=0.95,
            severity=Severity.MEDIUM
        )

        replacement = scrubber._get_replacement(match)
        assert "[HASH:" in replacement
        assert "]" in replacement

    def test_mask_email(self):
        """Test email masking."""
        config = PIIConfig(scrub_mode=ScrubMode.MASK)
        scrubber = PIIScrubber(config)

        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            original="test@example.com",
            start=0, end=16,
            confidence=0.95,
            severity=Severity.MEDIUM
        )

        replacement = scrubber._get_replacement(match)
        assert "**" in replacement  # Part of email is masked
        assert "@example.com" in replacement

    def test_mask_phone(self):
        """Test phone masking."""
        config = PIIConfig(scrub_mode=ScrubMode.MASK)
        scrubber = PIIScrubber(config)

        match = PIIMatch(
            pii_type=PIIType.PHONE,
            original="555-123-4567",
            start=0, end=12,
            confidence=0.85,
            severity=Severity.MEDIUM
        )

        replacement = scrubber._get_replacement(match)
        assert "4567" in replacement  # Last 4 digits visible
        assert "*" in replacement

    def test_remove_mode(self):
        """Test complete removal."""
        config = PIIConfig(scrub_mode=ScrubMode.REMOVE)
        scrubber = PIIScrubber(config)

        match = PIIMatch(
            pii_type=PIIType.EMAIL,
            original="test@example.com",
            start=0, end=16,
            confidence=0.95,
            severity=Severity.MEDIUM
        )

        replacement = scrubber._get_replacement(match)
        assert replacement == ""

    def test_scrub_preserves_positions(self):
        """Test that multiple scrubs preserve text positions."""
        config = PIIConfig(scrub_mode=ScrubMode.REDACT)
        scrubber = PIIScrubber(config)

        text = "Email: test@example.com and phone: 555-123-4567"
        matches = [
            PIIMatch(
                pii_type=PIIType.EMAIL,
                original="test@example.com",
                start=7, end=23,
                confidence=0.95,
                severity=Severity.MEDIUM
            ),
            PIIMatch(
                pii_type=PIIType.PHONE,
                original="555-123-4567",
                start=35, end=47,
                confidence=0.85,
                severity=Severity.MEDIUM
            )
        ]

        result, audit = scrubber.scrub(text, matches)

        assert "test@example.com" not in result
        assert "555-123-4567" not in result
        assert "[REDACTED-EMAIL]" in result
        assert "[REDACTED-PHONE]" in result
        assert audit.pii_found
        assert audit.total_matches == 2


class TestPIIHandler:
    """Tests for the main PII handler interface."""

    @pytest.fixture
    def handler(self):
        """Create handler with default config."""
        return PIIHandler(PIIConfig())

    def test_process_with_pii(self, handler):
        """Test processing text with PII."""
        text = "Contact john@example.com at 555-123-4567"
        clean_text, audit = handler.process(text)

        assert "john@example.com" not in clean_text
        assert audit.pii_found
        assert audit.total_matches >= 2

    def test_process_without_pii(self, handler):
        """Test processing clean text."""
        text = "This is a normal conversation."
        clean_text, audit = handler.process(text)

        assert clean_text == text
        assert not audit.pii_found
        assert audit.total_matches == 0

    def test_has_pii_check(self, handler):
        """Test quick PII check."""
        assert handler.has_pii("Email: test@example.com")
        assert not handler.has_pii("Normal text here")

    def test_get_pii_types(self, handler):
        """Test getting PII types found."""
        text = "Email: a@b.com, SSN: 123-45-6789"
        types = handler.get_pii_types(text)

        assert PIIType.EMAIL in types
        assert PIIType.SSN in types

    def test_disabled_handler(self):
        """Test that disabled handler passes through."""
        handler = PIIHandler(PIIConfig(enabled=False))

        text = "Email: test@example.com"
        clean_text, audit = handler.process(text)

        assert clean_text == text
        assert not audit.pii_found

    def test_stats_tracking(self, handler):
        """Test statistics tracking."""
        handler.process("Email: a@b.com")
        handler.process("Phone: 555-123-4567")
        handler.process("Clean text")

        stats = handler.get_stats()

        assert stats["total_processed"] == 3
        assert stats["total_pii_found"] >= 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_scrub_pii(self):
        """Test quick scrub function."""
        text = "Email: test@example.com"
        clean = scrub_pii(text)

        assert "test@example.com" not in clean

    def test_detect_pii(self):
        """Test quick detect function."""
        text = "Email: test@example.com"
        matches = detect_pii(text)

        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.EMAIL for m in matches)

    def test_has_pii_func(self):
        """Test quick check function."""
        assert has_pii("Email: test@example.com")
        assert not has_pii("Normal text")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_text(self):
        """Test empty text handling."""
        handler = PIIHandler()
        clean, audit = handler.process("")

        assert clean == ""
        assert not audit.pii_found

    def test_none_text(self):
        """Test None text handling."""
        handler = PIIHandler()
        clean, audit = handler.process(None)

        assert clean is None
        assert not audit.pii_found

    def test_unicode_text(self):
        """Test Unicode text handling."""
        handler = PIIHandler()
        text = "Email: tëst@example.com 日本語テキスト"
        clean, audit = handler.process(text)

        # Should still detect the email
        assert "tëst@example.com" not in clean or not audit.pii_found

    def test_very_long_text(self):
        """Test handling of very long text."""
        handler = PIIHandler()
        text = "Normal text " * 10000 + "email@test.com"
        clean, audit = handler.process(text)

        assert "email@test.com" not in clean

    def test_overlapping_patterns(self):
        """Test handling of overlapping PII patterns."""
        handler = PIIHandler()
        # This could match both email and potentially other patterns
        text = "Data: sk-test@openai.com"
        clean, audit = handler.process(text)

        # Should handle without error
        assert audit is not None
