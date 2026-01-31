"""
Technician Agent - Walks the Decision Tree (Programmatic Logic)
NO LLM - Pure Python deterministic logic

This agent checks:
1. Pixel created (Floodlight_Activity_ID exists)
2. Pixel firing (Last_Conversion_Date recent, Cookie counts > 0)
3. GTM linked and IDs match
4. Counting methods match between DV360 and GTM
5. Network calls not blocked
6. GA4 connected and data matches
"""
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
import time


class TechnicianAgent:
    """Agent that walks the pixel decision tree programmatically."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.decision_tree = self._load_decision_tree()
        self.findings = []
        self.reasoning_steps = []
        
    def _load_decision_tree(self) -> dict:
        """Load the pixel decision tree JSON."""
        tree_path = self.data_dir / "pixel_decision_tree.json"
        with open(tree_path, 'r') as f:
            return json.load(f)
    
    def _load_data(self) -> dict:
        """Load all CSV data files."""
        return {
            'dv360': pd.read_csv(self.data_dir / "mock_dv360_audit_ready.csv"),
            'gtm': pd.read_csv(self.data_dir / "mock_gtm_audit_ready.csv"),
            'ga4': pd.read_csv(self.data_dir / "mock_ga4_audit_ready.csv"),
            'website': pd.read_csv(self.data_dir / "mock_website_scan_audit_ready.csv")
        }
    
    def _log_step(self, step: str) -> dict:
        """Log a reasoning step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": "Technician",
            "step": step
        }
        self.reasoning_steps.append(log_entry)
        return log_entry
    
    def run_audit(self, limit: int = None, batch_size: int = 20) -> Generator[dict, None, None]:
        """
        Run the full audit using the decision tree logic in batches to simulate live stream.
        Yields findings as they are discovered.
        """
        self.findings = []
        self.reasoning_steps = []
        data = self._load_data()
        
        yield self._log_step("🔍 Technician Agent starting audit...")
        
        dv360_df = data['dv360']
        gtm_df = data['gtm'] # Reference data (static)
        website_df = data['website']
        ga4_df = data['ga4']
        
        if limit:
            dv360_df = dv360_df.head(limit)
            website_df = website_df.head(limit)

        yield self._log_step(f"📊 Loaded {len(dv360_df)} DV360 records to analyze in batches of {batch_size}")

        # Create batches
        # We drive the main loop with DV360 data as "the stream"
        total_batches = (len(dv360_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # Slice the dataframes to simulate a "batch" of new data arriving
            current_dv360_batch = dv360_df.iloc[start_idx:end_idx]
            current_website_batch = website_df.iloc[start_idx:end_idx] if start_idx < len(website_df) else pd.DataFrame()
            
            yield {
                "type": "batch_start", 
                "agent": "Technician",
                "batch_id": batch_idx + 1, 
                "total_batches": total_batches,
                "size": len(current_dv360_batch)
            }
            yield self._log_step(f"📦 Processing Batch {batch_idx + 1}/{total_batches} ({len(current_dv360_batch)} records)...")
        
            # Check 1: Pixel Created?
            for finding in self._check_pixel_created(current_dv360_batch): yield finding
            
            # Check 2: Pixel Firing?
            for finding in self._check_pixel_firing(current_dv360_batch): yield finding
            
            # Check 3: GTM Linked?
            for finding in self._check_gtm_linkage(current_dv360_batch, gtm_df): yield finding
            
            # Check 4: Counting Methods?
            for finding in self._check_counting_methods(current_dv360_batch, gtm_df): yield finding
            
            # Check 5: Network Calls Blocked?
            if not current_website_batch.empty:
               for finding in self._check_network_blocked(current_website_batch): yield finding

            # Check 7: GA4 Data Discrepancy? (Moved inside loop)
            for finding in self._check_ga4_discrepancy(current_dv360_batch, ga4_df): yield finding
            
            yield {"type": "batch_complete", "batch_id": batch_idx + 1}
            time.sleep(0.5)
            
        # Check 6: Consent Settings? (Static Check)
        yield self._log_step("🔎 CHECK 6: Verifying consent settings in GTM (Static Config)...")
        for finding in self._check_consent_settings(gtm_df):
            yield finding
        
        yield self._log_step(f"✅ Technician Agent completed. Found {len(self.findings)} issues.")
    def _check_pixel_created(self, dv360_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if Floodlight pixels are created (not nan)."""
        missing_pixels = dv360_df[dv360_df['Floodlight_Activity_ID'].isna() | 
                                   (dv360_df['Floodlight_Activity_ID'] == 'nan')]
        
        for _, row in missing_pixels.iterrows():
            finding = {
                "agent": "Technician",
                "check": "Pixel Created",
                "priority": "P0",
                "priority_label": "CRITICAL",
                "issue": "Missing Floodlight Pixel",
                "advertiser_id": row['Advertiser_ID'],
                "line_item": row['Line_Item_ID'],
                "daily_spend": row['Daily_Spend'],
                "technical_proof": f"Floodlight_Activity_ID is null/missing",
                "reasoning": [
                    f"Line Item {row['Line_Item_ID']} is ACTIVE and spending ${row['Daily_Spend']:.2f}/day",
                    "No Floodlight pixel is configured for this line item",
                    "Without a pixel, Google cannot track conversions",
                    "The bidding algorithm is optimizing towards NOTHING"
                ],
                "recommendation": "Create a Floodlight activity in DV360 and link it to this line item"
            }
            self.findings.append(finding)
            yield {"type": "finding", "data": finding}
    
    def _check_pixel_firing(self, dv360_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if pixels are actually firing (recent conversions, cookie consent > 0)."""
        today = datetime.now().date()
        
        for _, row in dv360_df.iterrows():
            if pd.isna(row['Floodlight_Activity_ID']) or row['Floodlight_Activity_ID'] == 'nan':
                continue  # Already caught in check 1
            
            # Check for stale conversions (> 7 days old)
            try:
                last_conv = datetime.strptime(str(row['Last_Conversion_Date']), '%Y-%m-%d').date()
                days_since = (today - last_conv).days
                
                if days_since > 7:
                    finding = {
                        "agent": "Technician",
                        "check": "Pixel Firing",
                        "priority": "P0",
                        "priority_label": "CRITICAL",
                        "issue": "Dead Pixel - No Recent Conversions",
                        "advertiser_id": row['Advertiser_ID'],
                        "floodlight_id": row['Floodlight_Activity_ID'],
                        "daily_spend": row['Daily_Spend'],
                        "days_since_conversion": days_since,
                        "technical_proof": f"Last_Conversion_Date = {row['Last_Conversion_Date']} ({days_since} days ago)",
                        "reasoning": [
                            f"Floodlight {row['Floodlight_Activity_ID']} last fired {days_since} days ago",
                            f"Daily spend is ${row['Daily_Spend']:.2f}",
                            f"Estimated wasted spend: ${row['Daily_Spend'] * days_since:.2f}",
                            "The algorithm is optimizing towards a dead signal"
                        ],
                        "recommendation": "Check if the pixel is properly placed on the conversion page"
                    }
                    self.findings.append(finding)
                    yield {"type": "finding", "data": finding}
            except:
                pass
            
            # Check for zero cookie consent
            if row['Cookie_Consented_Count'] == 0 and row['Cookie_Unconsented_Count'] == 0:
                finding = {
                    "agent": "Technician",
                    "check": "Cookie Consent",
                    "priority": "P0",
                    "priority_label": "CRITICAL",
                    "issue": "Cookie Consent Blocking All Data",
                    "advertiser_id": row['Advertiser_ID'],
                    "floodlight_id": row['Floodlight_Activity_ID'],
                    "daily_spend": row['Daily_Spend'],
                    "technical_proof": f"Cookie_Consented_Count = 0, Cookie_Unconsented_Count = 0",
                    "reasoning": [
                        "Both consented and unconsented cookie counts are ZERO",
                        "This means the cookie banner is blocking ALL data collection",
                        f"100% of the ${row['Daily_Spend']:.2f}/day spend has no attribution",
                        "The bidding algorithm cannot learn anything"
                    ],
                    "recommendation": "Review cookie consent implementation - ensure Consent Mode v2 is properly configured"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
    
    def _check_gtm_linkage(self, dv360_df: pd.DataFrame, gtm_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if GTM is linked and Advertiser IDs match."""
        for _, dv_row in dv360_df.iterrows():
            if pd.isna(dv_row['Floodlight_Activity_ID']) or dv_row['Floodlight_Activity_ID'] == 'nan':
                continue
            
            gtm_container = dv_row['GTM_Container_Link']
            
            # Find matching GTM tags
            matching_gtm = gtm_df[
                (gtm_df['Container_ID'] == gtm_container) & 
                (gtm_df['Linked_Floodlight_ID'] == dv_row['Floodlight_Activity_ID'])
            ]
            
            if len(matching_gtm) == 0:
                continue  # No matching tag found
            
            # Check for Advertiser ID mismatch
            for _, gtm_row in matching_gtm.iterrows():
                if gtm_row['Advertiser_ID_Config'] == 'ADV_MISMATCH' or \
                   gtm_row['Advertiser_ID_Config'] != dv_row['Advertiser_ID']:
                    finding = {
                        "agent": "Technician",
                        "check": "Advertiser ID Match",
                        "priority": "P1",
                        "priority_label": "HIGH",
                        "issue": "Advertiser ID Mismatch Between GTM and DV360",
                        "advertiser_id": dv_row['Advertiser_ID'],
                        "gtm_advertiser_id": gtm_row['Advertiser_ID_Config'],
                        "floodlight_id": dv_row['Floodlight_Activity_ID'],
                        "daily_spend": dv_row['Daily_Spend'],
                        "technical_proof": f"DV360 Advertiser: {dv_row['Advertiser_ID']} != GTM Config: {gtm_row['Advertiser_ID_Config']}",
                        "reasoning": [
                            f"GTM tag {gtm_row['Tag_ID']} has Advertiser ID: {gtm_row['Advertiser_ID_Config']}",
                            f"DV360 Floodlight expects Advertiser ID: {dv_row['Advertiser_ID']}",
                            "This mismatch causes attribution to fail silently",
                            "Conversions are being recorded but not attributed correctly"
                        ],
                        "recommendation": f"Update GTM tag to use Advertiser ID: {dv_row['Advertiser_ID']}"
                    }
                    self.findings.append(finding)
                    yield {"type": "finding", "data": finding}
    
    def _check_counting_methods(self, dv360_df: pd.DataFrame, gtm_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check if counting methods match between DV360 and GTM."""
        for _, dv_row in dv360_df.iterrows():
            if pd.isna(dv_row['Floodlight_Activity_ID']) or dv_row['Floodlight_Activity_ID'] == 'nan':
                continue
            
            gtm_container = dv_row['GTM_Container_Link']
            
            matching_gtm = gtm_df[
                (gtm_df['Container_ID'] == gtm_container) & 
                (gtm_df['Linked_Floodlight_ID'] == dv_row['Floodlight_Activity_ID'])
            ]
            
            for _, gtm_row in matching_gtm.iterrows():
                dv_method = str(dv_row['Counting_Method']).lower()
                gtm_method = str(gtm_row['Configured_Counting_Method']).lower()
                
                if dv_method != gtm_method and gtm_method != 'nan':
                    finding = {
                        "agent": "Technician",
                        "check": "Counting Method Match",
                        "priority": "P1",
                        "priority_label": "HIGH",
                        "issue": "Counting Method Mismatch",
                        "advertiser_id": dv_row['Advertiser_ID'],
                        "floodlight_id": dv_row['Floodlight_Activity_ID'],
                        "dv360_method": dv_row['Counting_Method'],
                        "gtm_method": gtm_row['Configured_Counting_Method'],
                        "daily_spend": dv_row['Daily_Spend'],
                        "technical_proof": f"DV360: {dv_row['Counting_Method']} != GTM: {gtm_row['Configured_Counting_Method']}",
                        "reasoning": [
                            f"DV360 expects counting method: {dv_row['Counting_Method']}",
                            f"GTM is configured with: {gtm_row['Configured_Counting_Method']}",
                            "This causes conversion counts to differ between platforms",
                            "Reporting will be inconsistent and unreliable"
                        ],
                        "recommendation": f"Align counting method in GTM to match DV360: {dv_row['Counting_Method']}"
                    }
                    self.findings.append(finding)
                    yield {"type": "finding", "data": finding}
    
    def _check_network_blocked(self, website_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check for blocked network calls on websites."""
        blocked = website_df[website_df['Network_Call_Status'] == '403 BLOCKED']
        
        for _, row in blocked.iterrows():
            finding = {
                "agent": "Technician",
                "check": "Network Call Status",
                "priority": "P0",
                "priority_label": "CRITICAL",
                "issue": "Floodlight Network Call Blocked",
                "url": row['URL'],
                "gtm_container": row['GTM_Container_Found'],
                "technical_proof": f"Network_Call_Status = 403 BLOCKED",
                "reasoning": [
                    f"The website {row['URL']} is blocking Floodlight network calls",
                    "This is typically caused by CSP headers or firewall rules",
                    "The pixel tag exists but cannot send data to Google",
                    "100% of conversions from this page are lost"
                ],
                "recommendation": "Ask the advertiser to whitelist fls.doubleclick.net in their CSP/firewall"
            }
            self.findings.append(finding)
            yield {"type": "finding", "data": finding}
    
    def _check_consent_settings(self, gtm_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check for missing consent settings in GTM."""
        missing_consent = gtm_df[gtm_df['Consent_Settings'].isna() | (gtm_df['Consent_Settings'] == 'nan')]
        
        for _, row in missing_consent.iterrows():
            finding = {
                "agent": "Technician",
                "check": "Consent Settings",
                "priority": "P1",
                "priority_label": "HIGH",
                "issue": "Missing Consent Settings in GTM Tag",
                "tag_id": row['Tag_ID'],
                "container_id": row['Container_ID'],
                "floodlight_id": row['Linked_Floodlight_ID'],
                "technical_proof": f"Consent_Settings is null/missing for tag {row['Tag_ID']}",
                "reasoning": [
                    f"GTM tag {row['Tag_ID']} has no consent settings configured",
                    "Without consent settings, the tag may fire regardless of user consent",
                    "This is a GDPR/privacy compliance risk",
                    "Could result in regulatory fines or account suspension"
                ],
                "recommendation": "Add consent settings (ad_storage, analytics_storage) to the tag configuration"
            }
            self.findings.append(finding)
            yield {"type": "finding", "data": finding}
    
    def _check_ga4_discrepancy(self, dv360_df: pd.DataFrame, ga4_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Check for large discrepancies between DV360 clicks and GA4 sessions."""
        # Get total clicks from DV360
        total_clicks = dv360_df['Clicks_Last_24h'].sum()
        total_sessions = ga4_df['Sessions_Last_24h'].sum()
        
        if total_clicks > 0:
            discrepancy = abs(total_clicks - total_sessions) / total_clicks * 100
            
            if discrepancy > 25:
                finding = {
                    "agent": "Technician",
                    "check": "DV360 vs GA4 Discrepancy",
                    "priority": "P1",
                    "priority_label": "HIGH",
                    "issue": "Significant Data Discrepancy Between DV360 and GA4",
                    "dv360_clicks": int(total_clicks),
                    "ga4_sessions": int(total_sessions),
                    "discrepancy_percent": round(discrepancy, 1),
                    "technical_proof": f"DV360 Clicks: {int(total_clicks)} vs GA4 Sessions: {int(total_sessions)} ({discrepancy:.1f}% difference)",
                    "reasoning": [
                        f"DV360 recorded {int(total_clicks)} clicks in the last 24 hours",
                        f"GA4 recorded only {int(total_sessions)} sessions",
                        f"This is a {discrepancy:.1f}% discrepancy (threshold: 25%)",
                        "Possible causes: UTM stripping, cross-domain issues, or duplicate pixels"
                    ],
                    "recommendation": "Check for cross-domain tracking issues or duplicate pixel placements"
                }
                self.findings.append(finding)
                yield {"type": "finding", "data": finding}
    
    def get_summary(self) -> dict:
        """Get a summary of all findings."""
        p0_count = len([f for f in self.findings if f['priority'] == 'P0'])
        p1_count = len([f for f in self.findings if f['priority'] == 'P1'])
        p2_count = len([f for f in self.findings if f['priority'] == 'P2'])
        
        return {
            "agent": "Technician",
            "total_findings": len(self.findings),
            "p0_critical": p0_count,
            "p1_high": p1_count,
            "p2_medium": p2_count,
            "findings": self.findings,
            "reasoning_steps": self.reasoning_steps
        }


if __name__ == "__main__":
    # Test the agent
    agent = TechnicianAgent()
    for event in agent.run_audit(limit=100):
        if event.get("type") == "finding":
            print(f"🚨 {event['data']['priority']}: {event['data']['issue']}")
        else:
            print(event.get("step", ""))
    
    summary = agent.get_summary()
    print(f"\n📊 Summary: {summary['total_findings']} issues found")
    print(f"   🔴 P0 Critical: {summary['p0_critical']}")
    print(f"   🟡 P1 High: {summary['p1_high']}")
    print(f"   🟢 P2 Medium: {summary['p2_medium']}")
