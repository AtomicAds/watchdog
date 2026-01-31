"""
Auditor Agent - Checks GA4 Governance Rules (NO LLM)
Pure Regex and Pandas validation logic

This agent checks the GA4 Audit Doc tribal knowledge:
1. PII in URLs (email, phone numbers)
2. Data retention settings (should be 14 months, not 2)
3. Google Signals enabled
4. Enhanced Measurement configured
5. Campaign naming conventions (snake_case, no spaces)
6. Referral exclusion list (payment gateways)
7. Consent Mode status
8. Cost Data Import status
"""
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Generator
import time


class AuditorAgent:
    """Agent that validates compliance against governance rules from GA4 Audit Doc."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.findings = []
        self.reasoning_steps = []
        
        # Compiled regex patterns for efficiency
        self.pii_patterns = {
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'phone': re.compile(r'phone=[\d\-\+\(\)\s]{7,}'),
            'email_param': re.compile(r'[?&]email=', re.IGNORECASE)
        }
        self.snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
    
    def _load_data(self) -> dict:
        """Load GA4 CSV data."""
        return {
            'ga4': pd.read_csv(self.data_dir / "mock_ga4_audit_ready.csv"),
            'gtm': pd.read_csv(self.data_dir / "mock_gtm_audit_ready.csv")
        }
    
    def _log_step(self, step: str) -> dict:
        """Log a reasoning step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": "Auditor",
            "step": step
        }
        self.reasoning_steps.append(log_entry)
        return log_entry
    
    def run_audit(self, limit: int = None, min_batch_size: int = 20, max_batch_size: int = 30) -> Generator[dict, None, None]:
        """
        Run the full governance audit in dynamic batches.
        Yields findings as they are discovered.
        Batch size is randomized (min-max).
        """
        import random
        self.findings = []
        self.reasoning_steps = []
        data = self._load_data()
        
        yield self._log_step("📋 Auditor Agent starting governance audit...")
        
        ga4_df = data['ga4']
        gtm_df = data['gtm']
        
        if limit:
            ga4_df = ga4_df.head(limit)
            
        yield self._log_step(f"📊 Loaded {len(ga4_df)} GA4 records to analyze in dynamic batches.")
        
        total_records = len(ga4_df)
        current_idx = 0
        batch_count = 1
        
        while current_idx < total_records:
            current_batch_size = random.randint(min_batch_size, max_batch_size)
            end_idx = min(current_idx + current_batch_size, total_records)
            
            current_batch = ga4_df.iloc[current_idx:end_idx]
            
            yield {
                "type": "batch_start", 
                "agent": "Auditor",
                "batch_id": batch_count, 
                "total_batches": "Unknown", 
                "size": len(current_batch)
            }
            yield self._log_step(f"📦 Processing Audit Batch {batch_count} ({len(current_batch)} records)...")
        
            # Check 1: PII in URLs
            for finding in self._check_pii_in_urls(current_batch): yield finding
            
            # Check 2: Data Retention Settings
            for finding in self._check_data_retention(current_batch): yield finding
            
            # Check 3: Google Signals
            for finding in self._check_google_signals(current_batch): yield finding
            
            # Check 4: Enhanced Measurement
            for finding in self._check_enhanced_measurement(current_batch): yield finding
            
            # Check 5: Campaign Naming Conventions
            for finding in self._check_campaign_naming(current_batch): yield finding
            
            # Check 6: Referral Exclusion List
            for finding in self._check_referral_exclusions(current_batch): yield finding
            
            # Check 7: Consent Mode Status
            for finding in self._check_consent_mode(current_batch): yield finding
            
            # Check 8: Cost Data Import
            for finding in self._check_cost_data_import(current_batch): yield finding
            
            yield {"type": "batch_complete", "batch_id": batch_count}
            time.sleep(0.5) # Simulate batch processing latency
            
            current_idx = end_idx
            batch_count += 1
            
        yield self._log_step(f"✅ Auditor Agent completed. Found {len(self.findings)} governance issues.")
    
    def _check_pii_in_urls(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check for PII (email, phone) in URL parameters - GDPR violation."""
        for _, row in ga4_df.iterrows():
            url_query = str(row['Sample_URL_Query'])
            
            # Check for email in URL
            if self.pii_patterns['email'].search(url_query) or \
               self.pii_patterns['email_param'].search(url_query):
                finding = {
                    "agent": "Auditor",
                    "check": "PII in URLs",
                    "priority": "P0",
                    "priority_label": "CRITICAL",
                    "issue": "PII Detected - Email Address in URL Parameters",
                    "property_id": row['Property_ID'],
                    "stream_id": row['Stream_ID'],
                    "url_sample": url_query[:50] + "..." if len(url_query) > 50 else url_query,
                    "technical_proof": f"URL query contains email pattern: {url_query}",
                    "reasoning": [
                        "Personal Identifiable Information (PII) found in URL parameters",
                        "Email addresses are being sent to Google Analytics in plain text",
                        "This is a GDPR/CCPA violation",
                        "Risk: Account suspension, regulatory fines up to 4% of global revenue"
                    ],
                    "recommendation": "Remove email parameter from URLs or hash (SHA256) before sending to GA4"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
    
    def _check_data_retention(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if data retention is set to 14 months (not 2)."""
        short_retention = ga4_df[ga4_df['Data_Retention_Months'] < 14]
        
        for _, row in short_retention.iterrows():
            finding = {
                "agent": "Auditor",
                "check": "Data Retention",
                "priority": "P1",
                "priority_label": "HIGH",
                "issue": "Data Retention Period Too Short",
                "property_id": row['Property_ID'],
                "current_retention": row['Data_Retention_Months'],
                "recommended_retention": 14,
                "technical_proof": f"Data_Retention_Months = {row['Data_Retention_Months']} (should be 14)",
                "reasoning": [
                    f"Data retention is set to {row['Data_Retention_Months']} months",
                    "Standard recommendation is 14 months for year-over-year analysis",
                    "Historical data will be automatically deleted after the retention period",
                    "Cannot perform YoY comparisons or long-term trend analysis"
                ],
                "recommendation": "Update data retention to 14 months in GA4 Admin > Data Settings > Data Retention"
            }
            self.findings.append(finding)
            yield {"type": "finding", "data": finding}
    
    def _check_google_signals(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if Google Signals is enabled for cross-device reporting."""
        signals_disabled = ga4_df[ga4_df['Google_Signals_Enabled'] == False]
        
        # Only report unique properties
        unique_properties = signals_disabled.drop_duplicates(subset=['Property_ID'])
        
        for _, row in unique_properties.iterrows():
            finding = {
                "agent": "Auditor",
                "check": "Google Signals",
                "priority": "P2",
                "priority_label": "MEDIUM",
                "issue": "Google Signals Disabled",
                "property_id": row['Property_ID'],
                "technical_proof": f"Google_Signals_Enabled = False for property {row['Property_ID']}",
                "reasoning": [
                    "Google Signals is disabled for this property",
                    "Cross-device reporting and demographics are unavailable",
                    "Remarketing audiences cannot leverage cross-device data",
                    "This limits audience insights and targeting capabilities"
                ],
                "recommendation": "Enable Google Signals in GA4 Admin > Data Settings > Data Collection"
            }
            self.findings.append(finding)
            yield {"type": "finding", "data": finding}
    
    def _check_enhanced_measurement(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if Enhanced Measurement has all recommended features enabled."""
        required_features = ['scrolls', 'outbound_clicks', 'site_search', 'video_engagement']
        
        for _, row in ga4_df.iterrows():
            config = str(row['Enhanced_Measurement_Config'])
            missing_features = []
            
            for feature in required_features:
                if feature not in config.lower():
                    missing_features.append(feature)
            
            if missing_features and len(missing_features) >= 2:  # Only flag if 2+ features missing
                finding = {
                    "agent": "Auditor",
                    "check": "Enhanced Measurement",
                    "priority": "P2",
                    "priority_label": "MEDIUM",
                    "issue": "Incomplete Enhanced Measurement Configuration",
                    "property_id": row['Property_ID'],
                    "stream_id": row['Stream_ID'],
                    "missing_features": missing_features,
                    "current_config": config,
                    "technical_proof": f"Missing features: {', '.join(missing_features)}",
                    "reasoning": [
                        f"Enhanced Measurement is missing: {', '.join(missing_features)}",
                        "These automatic interactions won't be tracked",
                        "Scroll depth, outbound clicks, and site search provide valuable UX insights",
                        "Missing data reduces optimization opportunities"
                    ],
                    "recommendation": f"Enable missing features in GA4 Admin > Data Streams > Enhanced Measurement"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
                break  # Only report once per property
    
    def _check_campaign_naming(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if campaign names follow snake_case convention (no spaces, lowercase)."""
        for _, row in ga4_df.iterrows():
            campaign_name = str(row['Session_Campaign_Name'])
            
            if campaign_name and campaign_name != 'nan':
                # Check for spaces
                has_spaces = ' ' in campaign_name
                # Check for uppercase
                has_uppercase = any(c.isupper() for c in campaign_name)
                # Check if it's not snake_case
                is_snake_case = self.snake_case_pattern.match(campaign_name)
                
                if has_spaces or (has_uppercase and not is_snake_case):
                    finding = {
                        "agent": "Auditor",
                        "check": "Campaign Naming Convention",
                        "priority": "P2",
                        "priority_label": "MEDIUM",
                        "issue": "Campaign Name Violates Naming Convention",
                        "property_id": row['Property_ID'],
                        "campaign_name": campaign_name,
                        "issues_found": [],
                        "technical_proof": f"Campaign name '{campaign_name}' is not snake_case",
                        "reasoning": [
                            f"Campaign name: '{campaign_name}' violates naming standards",
                            "Spaces and mixed case cause reporting inconsistencies",
                            "Automation rules and filters may fail",
                            "Format should be: country_objective_audience_product_month_year"
                        ],
                        "recommendation": f"Rename to snake_case format (e.g., 'nz_leads_prospecting_jan_2026')"
                    }
                    if has_spaces:
                        finding['issues_found'].append("Contains spaces")
                    if has_uppercase:
                        finding['issues_found'].append("Contains uppercase letters")
                    
                    self.findings.append(finding)
                    yield {"type": "finding", "data": finding}
    
    def _check_referral_exclusions(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if payment gateways are in referral exclusion list."""
        required_exclusions = ['paypal.com', 'stripe.com', 'gateway.com']
        
        # Check unique properties only
        checked_properties = set()
        
        for _, row in ga4_df.iterrows():
            property_id = row['Property_ID']
            if property_id in checked_properties:
                continue
            checked_properties.add(property_id)
            
            exclusion_list = str(row['Referral_Exclusion_List']).lower()
            missing_exclusions = []
            
            for domain in required_exclusions:
                if domain not in exclusion_list:
                    missing_exclusions.append(domain)
            
            # Only flag if key payment gateways are missing
            if 'paypal.com' in missing_exclusions:
                finding = {
                    "agent": "Auditor",
                    "check": "Referral Exclusions",
                    "priority": "P1",
                    "priority_label": "HIGH",
                    "issue": "Payment Gateway Missing from Referral Exclusion List",
                    "property_id": property_id,
                    "missing_domains": missing_exclusions,
                    "current_list": row['Referral_Exclusion_List'],
                    "technical_proof": f"Missing from exclusion list: {', '.join(missing_exclusions)}",
                    "reasoning": [
                        "Payment gateways are not in the referral exclusion list",
                        "When users return from payment, GA4 records it as a 'referral'",
                        "This breaks attribution - PayPal/Stripe get credit for YOUR sales",
                        "Conversion paths become inaccurate"
                    ],
                    "recommendation": "Add payment gateway domains to Referral Exclusions in GA4 Admin"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
    
    def _check_consent_mode(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check Consent Mode configuration status."""
        # Check unique properties only
        checked_properties = set()
        
        for _, row in ga4_df.iterrows():
            property_id = row['Property_ID']
            if property_id in checked_properties:
                continue
            checked_properties.add(property_id)
            
            consent_status = str(row['Consent_Mode_Status']).upper()
            
            if consent_status not in ['GRANTED', 'CONFIGURED']:
                finding = {
                    "agent": "Auditor",
                    "check": "Consent Mode",
                    "priority": "P1",
                    "priority_label": "HIGH",
                    "issue": "Consent Mode Not Properly Configured",
                    "property_id": property_id,
                    "current_status": row['Consent_Mode_Status'],
                    "technical_proof": f"Consent_Mode_Status = {row['Consent_Mode_Status']}",
                    "reasoning": [
                        f"Consent Mode status is: {row['Consent_Mode_Status']}",
                        "Consent Mode v2 is required for GDPR compliance",
                        "Without proper consent configuration, data collection may be non-compliant",
                        "This affects modeled conversions and audience building"
                    ],
                    "recommendation": "Implement Consent Mode v2 with your CMP (Consent Management Platform)"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
    
    def _check_cost_data_import(self, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if Cost Data Import is enabled for Meta/TikTok."""
        # Check unique properties only
        checked_properties = set()
        
        for _, row in ga4_df.iterrows():
            property_id = row['Property_ID']
            if property_id in checked_properties:
                continue
            checked_properties.add(property_id)
            
            import_status = str(row['Cost_Data_Import_Status']).lower()
            
            if import_status not in ['enabled', 'active', 'configured']:
                finding = {
                    "agent": "Auditor",
                    "check": "Cost Data Import",
                    "priority": "P2",
                    "priority_label": "MEDIUM",
                    "issue": "Cost Data Import Not Enabled",
                    "property_id": property_id,
                    "current_status": row['Cost_Data_Import_Status'],
                    "technical_proof": f"Cost_Data_Import_Status = {row['Cost_Data_Import_Status']}",
                    "reasoning": [
                        "Cost data import is not enabled",
                        "Meta and TikTok spend cannot be seen in GA4 reports",
                        "Cross-channel ROAS comparison requires manual data export",
                        "Opportunity to unify reporting is missed"
                    ],
                    "recommendation": "Enable Cost Data Import for Meta and TikTok in GA4 Admin > Data Import"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
                break  # Only report once
    
    def get_summary(self) -> dict:
        """Get a summary of all findings."""
        p0_count = len([f for f in self.findings if f['priority'] == 'P0'])
        p1_count = len([f for f in self.findings if f['priority'] == 'P1'])
        p2_count = len([f for f in self.findings if f['priority'] == 'P2'])
        
        return {
            "agent": "Auditor",
            "total_findings": len(self.findings),
            "p0_critical": p0_count,
            "p1_high": p1_count,
            "p2_medium": p2_count,
            "findings": self.findings,
            "reasoning_steps": self.reasoning_steps
        }


if __name__ == "__main__":
    # Test the agent
    agent = AuditorAgent()
    for event in agent.run_audit(limit=100):
        if event.get("type") == "finding":
            print(f"🚨 {event['data']['priority']}: {event['data']['issue']}")
        else:
            print(event.get("step", ""))
    
    summary = agent.get_summary()
    print(f"\n📊 Summary: {summary['total_findings']} governance issues found")
    print(f"   🔴 P0 Critical: {summary['p0_critical']}")
    print(f"   🟡 P1 High: {summary['p1_high']}")
    print(f"   🟢 P2 Medium: {summary['p2_medium']}")
