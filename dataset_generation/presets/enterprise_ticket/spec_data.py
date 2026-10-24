"""Enterprise-ticket schema and preset constants for structured dataset generation."""

from __future__ import annotations

import re
from typing import Any, TypedDict


class DifficultyProfileConfig(TypedDict):
    name: str
    writing_style_adjustments: dict[str, float]
    clarity_adjustments: dict[str, float]
    opener_probability: float
    explicit_system_probability: float
    explicit_resolution_probability: float
    context_probability: float
    progress_probability: float
    pressure_probability: float
    indirect_issue_probability: float
    symptom_first_probability: float
    sentence_pattern_weights: dict[str, float]


GRANULARITY_LABEL_COMPONENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "organizational": {
        "coarse": ("department",),
        "normal": ("department", "sub_department"),
        "fine": ("department", "sub_department", "owning_team"),
    },
    "intent": {
        "coarse": ("intent_group",),
        "normal": ("intent_group", "issue_type"),
        "fine": ("intent_group", "issue_type", "sub_intent"),
    },
}

GRANULARITY_LABEL_FIELDS: dict[str, dict[str, str]] = {
    "intent": {
        "coarse": "intent_label_coarse",
        "normal": "intent_label_normal",
        "fine": "intent_label_fine",
    },
    "organizational": {
        "coarse": "organizational_label_coarse",
        "normal": "organizational_label_normal",
        "fine": "organizational_label_fine",
    },
}


SCHEMA: dict[str, dict[str, Any]] = {
    "department": {
        "values": [
            "HR",
            "Finance",
            "IT",
            "Security",
            "Facilities",
            "Procurement",
            "Legal",
            "Sales",
            "Marketing",
            "Customer Support",
            "Engineering",
            "Data",
            "Travel",
            "Workplace Experience",
            "Payroll",
            "Regional Operations",
        ]
    },
    "sub_department": {
        "parents": ["department"],
        "mapping": {
            "HR": ["Payroll", "Benefits", "Onboarding", "Employee Relations"],
            "Finance": ["Accounts Payable", "Accounts Receivable", "FP&A", "Tax"],
            "IT": ["Help Desk", "IAM", "Network", "Enterprise Apps"],
            "Security": ["SecOps", "GRC", "Incident Response", "Identity Governance"],
            "Facilities": ["Maintenance", "Badge Access", "Workspace", "Office Services"],
            "Procurement": ["Vendor Management", "Purchase Requests", "Asset Procurement"],
            "Legal": ["Contract Review", "Privacy", "Compliance Support"],
            "Sales": ["Sales Operations", "Deal Desk", "Territory Planning"],
            "Marketing": ["Marketing Operations", "Demand Generation", "Events"],
            "Customer Support": [
                "Support Operations",
                "Escalations",
                "Knowledge Management",
            ],
            "Engineering": [
                "Developer Productivity",
                "Infrastructure",
                "Release Engineering",
            ],
            "Data": ["Analytics Engineering", "Business Intelligence", "Data Governance"],
            "Travel": ["Travel Desk", "Expense Operations", "Mobility"],
            "Workplace Experience": [
                "Office Services",
                "Events and Hospitality",
                "Visitor Services",
            ],
            "Payroll": [
                "Payroll Operations",
                "Tax Operations",
                "Compensation Administration",
            ],
            "Regional Operations": [
                "Site Operations",
                "Local Compliance",
                "Regional Services",
            ],
        },
    },
    "requester_role": {
        "values": [
            "employee",
            "manager",
            "director",
            "contractor",
            "intern",
            "executive_assistant",
            "new_hire",
        ]
    },
    "owning_team": {
        "parents": ["department"],
        "mapping": {
            "HR": ["HR Operations", "Benefits Administration", "People Services"],
            "Finance": [
                "Finance Operations",
                "Accounts Payable Operations",
                "Procurement Finance",
            ],
            "IT": ["IT Support", "IAM Team"],
            "Security": ["Security Team", "Identity Security", "Incident Response"],
            "Facilities": ["Facilities Team", "Badge Services", "Workplace Services"],
            "Procurement": ["Procurement Team", "Vendor Operations", "Purchasing Desk"],
            "Legal": ["Legal Ops", "Privacy Review", "Commercial Legal"],
            "Sales": ["Sales Operations", "Revenue Systems"],
            "Marketing": ["Marketing Operations", "MarTech Support"],
            "Customer Support": ["Support Operations", "Customer Systems"],
            "Engineering": ["Developer Productivity", "Release Engineering"],
            "Data": ["Data Platform", "Analytics Operations"],
            "Travel": ["Travel Operations", "Expense Audit"],
            "Workplace Experience": [
                "Workplace Experience Team",
                "Events Support",
                "Visitor Services",
            ],
            "Payroll": ["Payroll Operations", "Payroll Tax Support"],
            "Regional Operations": [
                "Regional Operations Team",
                "Site Services",
                "Regional Compliance",
            ],
        },
    },
    "ticket_category": {
        "parents": ["department"],
        "mapping": {
            "HR": [
                "data_update",
                "policy_question",
                "benefits_issue",
                "onboarding_task",
            ],
            "Finance": [
                "reimbursement",
                "invoice_issue",
                "vendor_setup",
                "purchase_request",
            ],
            "IT": ["access_request", "account_issue", "software_issue", "service_request"],
            "Security": ["security_concern", "access_review", "incident_report"],
            "Facilities": ["maintenance_request", "badge_issue", "workspace_request"],
            "Procurement": ["vendor_request", "purchase_approval", "contract_purchase"],
            "Legal": ["contract_review", "privacy_request", "policy_clarification"],
            "Sales": ["crm_support", "quote_operations", "territory_management"],
            "Marketing": ["campaign_operations", "marketing_tools", "event_marketing"],
            "Customer Support": [
                "support_platform",
                "case_operations",
                "customer_communications",
            ],
            "Engineering": ["developer_access", "developer_tooling", "release_operations"],
            "Data": ["analytics_access", "data_platform", "metrics_governance"],
            "Travel": ["travel_booking", "travel_expense", "traveler_support"],
            "Workplace Experience": ["office_services", "event_support", "visitor_services"],
            "Payroll": ["pay_cycle_issue", "tax_documents", "compensation_changes"],
            "Regional Operations": [
                "site_operations",
                "regional_compliance",
                "local_services",
            ],
        },
    },
    "issue_type": {
        "parents": ["sub_intent"],
        "mapping": {
            "manager_change": ["people_administration_issue"],
            "title_change": ["people_administration_issue"],
            "personal_info_correction": ["people_administration_issue"],
            "dependent_addition": ["people_administration_issue"],
            "enrollment_problem": ["people_administration_issue"],
            "coverage_question": ["people_administration_issue"],
            "leave_policy_question": ["policy_question"],
            "remote_work_policy_question": ["policy_question"],
            "employment_policy_question": ["policy_question"],
            "new_hire_setup": ["onboarding_process_issue"],
            "orientation_question": ["onboarding_process_issue"],
            "missing_onboarding_step": ["onboarding_process_issue"],
            "missing_reimbursement": ["finance_transaction_issue"],
            "duplicate_claim": ["finance_transaction_issue"],
            "receipt_followup": ["finance_transaction_issue"],
            "invoice_mismatch": ["finance_transaction_issue"],
            "payment_delay": ["finance_transaction_issue"],
            "duplicate_invoice": ["finance_transaction_issue"],
            "new_vendor_creation": ["procurement_workflow_issue"],
            "missing_tax_form": ["procurement_workflow_issue"],
            "bank_detail_update": ["procurement_workflow_issue"],
            "po_creation": ["procurement_workflow_issue"],
            "budget_code_question": ["procurement_workflow_issue"],
            "approval_missing": ["procurement_workflow_issue"],
            "vpn_access": ["access_request"],
            "shared_drive_access": ["access_request"],
            "admin_access": ["access_request"],
            "temporary_access": ["access_request"],
            "password_reset": ["authentication_issue"],
            "account_locked": ["authentication_issue"],
            "mfa_problem": ["authentication_issue"],
            "license_missing": ["service_issue"],
            "installation_failure": ["service_issue"],
            "application_crash": ["service_issue"],
            "laptop_setup": ["service_issue"],
            "software_installation": ["service_issue"],
            "hardware_replacement": ["service_issue"],
            "phishing_report": ["security_incident"],
            "suspicious_login": ["security_incident"],
            "data_exposure_concern": ["security_incident"],
            "privileged_access_review": ["access_request"],
            "terminated_user_access_check": ["access_request"],
            "role_audit": ["access_request"],
            "malware_alert": ["security_incident"],
            "policy_violation": ["security_incident"],
            "credential_compromise": ["security_incident"],
            "ac_issue": ["workplace_support_issue"],
            "lighting_problem": ["workplace_support_issue"],
            "cleaning_issue": ["workplace_support_issue"],
            "badge_not_working": ["access_request"],
            "new_badge_request": ["access_request"],
            "visitor_access": ["access_request"],
            "desk_move": ["workplace_support_issue"],
            "meeting_room_setup": ["workplace_support_issue"],
            "ergonomic_equipment": ["workplace_support_issue"],
            "new_supplier_request": ["procurement_workflow_issue"],
            "vendor_update": ["procurement_workflow_issue"],
            "vendor_document_missing": ["procurement_workflow_issue"],
            "software_purchase": ["procurement_workflow_issue"],
            "equipment_purchase": ["procurement_workflow_issue"],
            "urgent_purchase_exception": ["procurement_workflow_issue"],
            "subscription_renewal": ["procurement_workflow_issue"],
            "service_contract_request": ["procurement_workflow_issue"],
            "quote_review": ["procurement_workflow_issue"],
            "nda_review": ["legal_review_issue"],
            "msa_review": ["legal_review_issue"],
            "vendor_contract_review": ["legal_review_issue"],
            "data_deletion_request": ["legal_review_issue"],
            "data_access_request": ["legal_review_issue"],
            "privacy_clause_question": ["legal_review_issue"],
            "policy_exception_question": ["policy_question"],
            "retention_policy_question": ["policy_question"],
            "legal_process_question": ["policy_question"],
            "crm_access_change": ["access_request"],
            "lead_routing_issue": ["sales_platform_issue"],
            "account_ownership_correction": ["sales_operations_issue"],
            "pipeline_visibility_issue": ["sales_platform_issue"],
            "quote_approval_followup": ["sales_operations_issue"],
            "quote_builder_sync_issue": ["sales_platform_issue"],
            "discount_exception_request": ["sales_operations_issue"],
            "quote_template_problem": ["sales_platform_issue"],
            "territory_update_request": ["sales_operations_issue"],
            "book_transfer_request": ["sales_operations_issue"],
            "territory_rule_mismatch": ["sales_platform_issue"],
            "account_coverage_gap": ["sales_operations_issue"],
            "campaign_launch_change": ["marketing_operations_issue"],
            "form_routing_issue": ["marketing_platform_issue"],
            "campaign_schedule_update": ["marketing_operations_issue"],
            "suppression_list_issue": ["marketing_platform_issue"],
            "marketing_tool_access": ["access_request"],
            "audience_sync_problem": ["marketing_platform_issue"],
            "nurture_enrollment_problem": ["marketing_platform_issue"],
            "marketing_permission_change": ["access_request"],
            "webinar_setup_issue": ["marketing_operations_issue"],
            "swag_vendor_followup": ["marketing_operations_issue"],
            "landing_page_publish_issue": ["marketing_platform_issue"],
            "asset_approval_delay": ["marketing_operations_issue"],
            "queue_access_change": ["access_request"],
            "macro_or_view_missing": ["customer_support_platform_issue"],
            "trigger_automation_issue": ["customer_support_platform_issue"],
            "agent_provisioning_request": ["access_request"],
            "case_reassignment_request": ["customer_support_operations_issue"],
            "handoff_queue_issue": ["customer_support_operations_issue"],
            "escalation_path_change": ["customer_support_operations_issue"],
            "case_merge_split_problem": ["customer_support_platform_issue"],
            "status_page_update_request": ["customer_support_operations_issue"],
            "help_center_publish_issue": ["customer_support_platform_issue"],
            "email_template_update": ["customer_support_operations_issue"],
            "outbound_notification_failure": ["customer_support_platform_issue"],
            "repo_access_request": ["access_request"],
            "cloud_role_access": ["access_request"],
            "build_platform_access": ["access_request"],
            "service_account_access": ["access_request"],
            "ci_pipeline_issue": ["developer_tooling_issue"],
            "package_registry_problem": ["developer_tooling_issue"],
            "build_runner_capacity_issue": ["developer_tooling_issue"],
            "environment_config_problem": ["developer_tooling_issue"],
            "release_window_question": ["release_operations_issue"],
            "feature_flag_access": ["access_request"],
            "deployment_freeze_exception": ["release_operations_issue"],
            "rollback_support_request": ["release_operations_issue"],
            "dashboard_access_request": ["access_request"],
            "warehouse_role_access": ["access_request"],
            "bi_group_membership_change": ["access_request"],
            "row_level_access_issue": ["access_request"],
            "pipeline_failure_report": ["data_platform_issue"],
            "scheduled_report_missing": ["data_platform_issue"],
            "dataset_backfill_request": ["data_platform_issue"],
            "stale_dataset_alert": ["data_platform_issue"],
            "metric_definition_question": ["metrics_governance_issue"],
            "tracking_gap_report": ["data_platform_issue"],
            "dashboard_discrepancy_report": ["metrics_governance_issue"],
            "source_of_truth_conflict": ["metrics_governance_issue"],
            "trip_booking_change": ["travel_operations_issue"],
            "unused_ticket_credit_question": ["travel_operations_issue"],
            "trip_approval_delay": ["travel_operations_issue"],
            "traveler_profile_sync_issue": ["travel_operations_issue"],
            "travel_expense_followup": ["travel_operations_issue"],
            "corporate_card_travel_issue": ["travel_operations_issue"],
            "per_diem_exception_question": ["travel_operations_issue"],
            "receipt_match_issue": ["travel_operations_issue"],
            "visa_letter_request": ["travel_operations_issue"],
            "emergency_travel_change": ["travel_operations_issue"],
            "travel_policy_exception": ["travel_operations_issue"],
            "traveler_profile_update": ["travel_operations_issue"],
            "office_supply_restock": ["workplace_experience_issue"],
            "catering_setup_issue": ["workplace_experience_issue"],
            "signage_update_request": ["workplace_experience_issue"],
            "pantry_service_issue": ["workplace_experience_issue"],
            "all_hands_room_reset": ["workplace_experience_issue"],
            "onsite_event_support": ["workplace_experience_issue"],
            "av_setup_issue": ["workplace_experience_issue"],
            "event_schedule_change": ["workplace_experience_issue"],
            "visitor_hosting_request": ["workplace_experience_issue"],
            "after_hours_access_setup": ["access_request"],
            "visitor_list_sync_issue": ["workplace_experience_issue"],
            "parking_access_request": ["access_request"],
            "missing_pay": ["payroll_operations_issue"],
            "timesheet_correction": ["payroll_operations_issue"],
            "overtime_calculation_issue": ["payroll_operations_issue"],
            "deduction_mismatch": ["payroll_operations_issue"],
            "w2_correction_request": ["payroll_operations_issue"],
            "tax_withholding_change": ["payroll_operations_issue"],
            "tax_document_delivery_issue": ["payroll_operations_issue"],
            "residency_tax_question": ["payroll_operations_issue"],
            "bonus_payment_followup": ["payroll_operations_issue"],
            "pay_stub_request": ["payroll_operations_issue"],
            "commission_payment_followup": ["payroll_operations_issue"],
            "retro_pay_adjustment_request": ["payroll_operations_issue"],
            "local_site_access_issue": ["access_request"],
            "regional_move_coordination": ["regional_operations_issue"],
            "site_readiness_check": ["regional_operations_issue"],
            "regional_seating_change": ["regional_operations_issue"],
            "country_process_question": ["regional_operations_issue"],
            "local_notice_update": ["regional_operations_issue"],
            "regional_audit_evidence_request": ["regional_operations_issue"],
            "policy_translation_update": ["regional_operations_issue"],
            "mailroom_request": ["regional_operations_issue"],
            "local_vendor_followup": ["regional_operations_issue"],
            "courier_delivery_issue": ["regional_operations_issue"],
            "reception_service_request": ["regional_operations_issue"],
        },
    },
    "intent_group": {
        "parents": ["issue_type"],
        "mapping": {
            "people_administration_issue": ["people_ops"],
            "onboarding_process_issue": ["people_ops"],
            "policy_question": ["policy_compliance"],
            "access_request": ["access_identity"],
            "authentication_issue": ["access_identity"],
            "service_issue": ["tech_support"],
            "finance_transaction_issue": ["finance_procurement"],
            "procurement_workflow_issue": ["finance_procurement"],
            "security_incident": ["security_operations"],
            "workplace_support_issue": ["workplace_operations"],
            "legal_review_issue": ["policy_compliance"],
            "sales_platform_issue": ["go_to_market_ops"],
            "sales_operations_issue": ["go_to_market_ops"],
            "marketing_platform_issue": ["go_to_market_ops"],
            "marketing_operations_issue": ["go_to_market_ops"],
            "customer_support_platform_issue": ["customer_operations"],
            "customer_support_operations_issue": ["customer_operations"],
            "developer_tooling_issue": ["engineering_operations"],
            "release_operations_issue": ["engineering_operations"],
            "data_platform_issue": ["data_operations"],
            "metrics_governance_issue": ["data_operations"],
            "travel_operations_issue": ["travel_operations"],
            "workplace_experience_issue": ["workplace_operations"],
            "payroll_operations_issue": ["people_ops"],
            "regional_operations_issue": ["workplace_operations"],
        },
    },
    "sub_intent": {
        "parents": ["department", "ticket_category"],
        "mapping": {
            ("HR", "data_update"): [
                "manager_change",
                "title_change",
                "personal_info_correction",
            ],
            ("HR", "benefits_issue"): [
                "dependent_addition",
                "enrollment_problem",
                "coverage_question",
            ],
            ("HR", "policy_question"): [
                "leave_policy_question",
                "remote_work_policy_question",
                "employment_policy_question",
            ],
            ("HR", "onboarding_task"): [
                "new_hire_setup",
                "orientation_question",
                "missing_onboarding_step",
            ],
            ("Finance", "reimbursement"): [
                "missing_reimbursement",
                "duplicate_claim",
                "receipt_followup",
            ],
            ("Finance", "invoice_issue"): [
                "invoice_mismatch",
                "payment_delay",
                "duplicate_invoice",
            ],
            ("Finance", "vendor_setup"): [
                "new_vendor_creation",
                "missing_tax_form",
                "bank_detail_update",
            ],
            ("Finance", "purchase_request"): [
                "po_creation",
                "budget_code_question",
                "approval_missing",
            ],
            ("IT", "access_request"): [
                "vpn_access",
                "shared_drive_access",
                "admin_access",
                "temporary_access",
            ],
            ("IT", "account_issue"): [
                "password_reset",
                "account_locked",
                "mfa_problem",
            ],
            ("IT", "software_issue"): [
                "license_missing",
                "installation_failure",
                "application_crash",
            ],
            ("IT", "service_request"): [
                "laptop_setup",
                "software_installation",
                "hardware_replacement",
            ],
            ("Security", "security_concern"): [
                "phishing_report",
                "suspicious_login",
                "data_exposure_concern",
            ],
            ("Security", "access_review"): [
                "privileged_access_review",
                "terminated_user_access_check",
                "role_audit",
            ],
            ("Security", "incident_report"): [
                "malware_alert",
                "policy_violation",
                "credential_compromise",
            ],
            ("Facilities", "maintenance_request"): [
                "ac_issue",
                "lighting_problem",
                "cleaning_issue",
            ],
            ("Facilities", "badge_issue"): [
                "badge_not_working",
                "new_badge_request",
                "visitor_access",
            ],
            ("Facilities", "workspace_request"): [
                "desk_move",
                "meeting_room_setup",
                "ergonomic_equipment",
            ],
            ("Procurement", "vendor_request"): [
                "new_supplier_request",
                "vendor_update",
                "vendor_document_missing",
            ],
            ("Procurement", "purchase_approval"): [
                "software_purchase",
                "equipment_purchase",
                "urgent_purchase_exception",
            ],
            ("Procurement", "contract_purchase"): [
                "subscription_renewal",
                "service_contract_request",
                "quote_review",
            ],
            ("Legal", "contract_review"): [
                "nda_review",
                "msa_review",
                "vendor_contract_review",
            ],
            ("Legal", "privacy_request"): [
                "data_deletion_request",
                "data_access_request",
                "privacy_clause_question",
            ],
            ("Legal", "policy_clarification"): [
                "policy_exception_question",
                "retention_policy_question",
                "legal_process_question",
            ],
            ("Sales", "crm_support"): [
                "crm_access_change",
                "lead_routing_issue",
                "account_ownership_correction",
                "pipeline_visibility_issue",
            ],
            ("Sales", "quote_operations"): [
                "quote_approval_followup",
                "quote_builder_sync_issue",
                "discount_exception_request",
                "quote_template_problem",
            ],
            ("Sales", "territory_management"): [
                "territory_update_request",
                "book_transfer_request",
                "territory_rule_mismatch",
                "account_coverage_gap",
            ],
            ("Marketing", "campaign_operations"): [
                "campaign_launch_change",
                "form_routing_issue",
                "campaign_schedule_update",
                "suppression_list_issue",
            ],
            ("Marketing", "marketing_tools"): [
                "marketing_tool_access",
                "audience_sync_problem",
                "nurture_enrollment_problem",
                "marketing_permission_change",
            ],
            ("Marketing", "event_marketing"): [
                "webinar_setup_issue",
                "swag_vendor_followup",
                "landing_page_publish_issue",
                "asset_approval_delay",
            ],
            ("Customer Support", "support_platform"): [
                "queue_access_change",
                "macro_or_view_missing",
                "trigger_automation_issue",
                "agent_provisioning_request",
            ],
            ("Customer Support", "case_operations"): [
                "case_reassignment_request",
                "handoff_queue_issue",
                "escalation_path_change",
                "case_merge_split_problem",
            ],
            ("Customer Support", "customer_communications"): [
                "status_page_update_request",
                "help_center_publish_issue",
                "email_template_update",
                "outbound_notification_failure",
            ],
            ("Engineering", "developer_access"): [
                "repo_access_request",
                "cloud_role_access",
                "build_platform_access",
                "service_account_access",
            ],
            ("Engineering", "developer_tooling"): [
                "ci_pipeline_issue",
                "package_registry_problem",
                "build_runner_capacity_issue",
                "environment_config_problem",
            ],
            ("Engineering", "release_operations"): [
                "release_window_question",
                "feature_flag_access",
                "deployment_freeze_exception",
                "rollback_support_request",
            ],
            ("Data", "analytics_access"): [
                "dashboard_access_request",
                "warehouse_role_access",
                "bi_group_membership_change",
                "row_level_access_issue",
            ],
            ("Data", "data_platform"): [
                "pipeline_failure_report",
                "scheduled_report_missing",
                "dataset_backfill_request",
                "stale_dataset_alert",
            ],
            ("Data", "metrics_governance"): [
                "metric_definition_question",
                "tracking_gap_report",
                "dashboard_discrepancy_report",
                "source_of_truth_conflict",
            ],
            ("Travel", "travel_booking"): [
                "trip_booking_change",
                "unused_ticket_credit_question",
                "trip_approval_delay",
                "traveler_profile_sync_issue",
            ],
            ("Travel", "travel_expense"): [
                "travel_expense_followup",
                "corporate_card_travel_issue",
                "per_diem_exception_question",
                "receipt_match_issue",
            ],
            ("Travel", "traveler_support"): [
                "visa_letter_request",
                "emergency_travel_change",
                "travel_policy_exception",
                "traveler_profile_update",
            ],
            ("Workplace Experience", "office_services"): [
                "office_supply_restock",
                "catering_setup_issue",
                "signage_update_request",
                "pantry_service_issue",
            ],
            ("Workplace Experience", "event_support"): [
                "all_hands_room_reset",
                "onsite_event_support",
                "av_setup_issue",
                "event_schedule_change",
            ],
            ("Workplace Experience", "visitor_services"): [
                "visitor_hosting_request",
                "after_hours_access_setup",
                "visitor_list_sync_issue",
                "parking_access_request",
            ],
            ("Payroll", "pay_cycle_issue"): [
                "missing_pay",
                "timesheet_correction",
                "overtime_calculation_issue",
                "deduction_mismatch",
            ],
            ("Payroll", "tax_documents"): [
                "w2_correction_request",
                "tax_withholding_change",
                "tax_document_delivery_issue",
                "residency_tax_question",
            ],
            ("Payroll", "compensation_changes"): [
                "bonus_payment_followup",
                "pay_stub_request",
                "commission_payment_followup",
                "retro_pay_adjustment_request",
            ],
            ("Regional Operations", "site_operations"): [
                "local_site_access_issue",
                "regional_move_coordination",
                "site_readiness_check",
                "regional_seating_change",
            ],
            ("Regional Operations", "regional_compliance"): [
                "country_process_question",
                "local_notice_update",
                "regional_audit_evidence_request",
                "policy_translation_update",
            ],
            ("Regional Operations", "local_services"): [
                "mailroom_request",
                "local_vendor_followup",
                "courier_delivery_issue",
                "reception_service_request",
            ],
        },
    },
    "system_name": {
        "parents": ["department"],
        "mapping": {
            "HR": [
                "Workday",
                "Greenhouse",
                "Benefits Portal",
                "Learning Management System",
            ],
            "Finance": [
                "NetSuite",
                "SAP",
                "Coupa",
                "Expense Portal",
                "Billing Portal",
            ],
            "IT": [
                "Okta",
                "Google Workspace",
                "Microsoft 365",
                "Jira",
                "VPN",
                "MDM Platform",
                "Service Desk Portal",
            ],
            "Security": [
                "Okta",
                "SIEM",
                "EDR Platform",
                "Access Governance Tool",
                "Security Awareness Platform",
                "Vulnerability Management Platform",
            ],
            "Facilities": [
                "Badge System",
                "Room Booking Tool",
                "Workplace Portal",
                "Maintenance Ticketing System",
            ],
            "Procurement": [
                "Coupa",
                "SAP",
                "Vendor Portal",
                "Purchase Intake Portal",
            ],
            "Legal": [
                "Contract Lifecycle Tool",
                "Document Management System",
                "E-signature Platform",
                "Privacy Request Portal",
            ],
            "Sales": [
                "Salesforce",
                "Quote Builder",
                "Sales Engagement Platform",
                "Revenue Intelligence Platform",
                "Territory Planning Tool",
            ],
            "Marketing": [
                "Marketo",
                "HubSpot",
                "CMS Platform",
                "Webinar Platform",
                "Digital Asset Manager",
            ],
            "Customer Support": [
                "Zendesk",
                "Statuspage",
                "Help Center",
                "Support Messaging Platform",
                "Customer CRM",
            ],
            "Engineering": [
                "GitHub",
                "Cloud Console",
                "Build Platform",
                "Feature Flag Service",
                "Incident Management Platform",
                "Secrets Manager",
                "Artifact Registry",
            ],
            "Data": [
                "Looker",
                "Snowflake",
                "dbt Cloud",
                "Tracking Console",
                "Workflow Orchestrator",
                "Data Catalog",
                "Experimentation Platform",
            ],
            "Travel": [
                "Travel Portal",
                "Concur",
                "Corporate Card Portal",
                "Visa Support Portal",
            ],
            "Workplace Experience": [
                "Workplace Portal",
                "Visitor Management System",
                "Catering Portal",
                "Event Calendar",
                "AV Booking Tool",
                "Parking Management System",
            ],
            "Payroll": [
                "Payroll Portal",
                "Workday",
                "ADP",
                "Timekeeping System",
            ],
            "Regional Operations": [
                "Local Operations Portal",
                "Workplace Portal",
                "Badge System",
                "Courier Tracking Portal",
                "Regional Compliance Tracker",
            ],
        },
    },
    "business_event_context": {
        "values": [
            "routine_operation",
            "new_hire_onboarding",
            "employee_offboarding",
            "quarter_end_close",
            "audit_preparation",
            "org_restructure",
            "product_launch",
            "office_move",
        ]
    },
    "workflow_stage": {
        "values": [
            "new_submission",
            "triaged",
            "pending_requester_info",
            "pending_approval",
            "in_progress",
            "blocked",
            "escalated",
        ]
    },
    "approval_required": {
        "values": [
            "none",
            "manager_approval",
            "finance_approval",
            "security_approval",
            "multi_step_approval",
        ]
    },
    "dependency_type": {
        "values": [
            "none",
            "waiting_for_document",
            "waiting_for_manager",
            "waiting_for_vendor",
            "waiting_for_system_sync",
            "waiting_for_another_team",
        ]
    },
    "urgency": {"values": ["low", "medium", "high", "critical"]},
    "severity": {"values": ["minor", "moderate", "major", "business_critical"]},
    "impact_scope": {
        "values": [
            "single_user",
            "single_team",
            "single_department",
            "multi_department",
            "company_wide",
        ]
    },
    "writing_style": {
        "values": [
            "concise",
            "formal",
            "informal",
            "frustrated",
            "technical",
            "non_native_english",
        ]
    },
    "clarity_level": {"values": ["clear", "moderately_clear", "vague", "missing_some_details"]},
}


def _granularity_label_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError(
            "Granularity label parts must contain at least one alphanumeric character."
        )
    return normalized


def build_granularity_label(*parts: str) -> str:
    return "__".join(_granularity_label_token(part) for part in parts)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _organizational_granularity_values(level: str) -> list[str]:
    labels: list[str] = []
    departments = SCHEMA["department"]["values"]
    if level == "coarse":
        return [build_granularity_label(department) for department in departments]
    if level == "normal":
        for department in departments:
            for sub_department in SCHEMA["sub_department"]["mapping"][department]:
                labels.append(build_granularity_label(department, sub_department))
        return labels
    for department in departments:
        for sub_department in SCHEMA["sub_department"]["mapping"][department]:
            for owning_team in SCHEMA["owning_team"]["mapping"][department]:
                labels.append(build_granularity_label(department, sub_department, owning_team))
    return labels


def _intent_granularity_values(level: str) -> list[str]:
    labels: list[str] = []
    if level == "coarse":
        seen: set[str] = set()
        for intent_groups in SCHEMA["intent_group"]["mapping"].values():
            for intent_group in intent_groups:
                label = build_granularity_label(intent_group)
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        return labels

    for (_department, _ticket_category), sub_intents in SCHEMA["sub_intent"]["mapping"].items():
        for sub_intent in sub_intents:
            issue_type = SCHEMA["issue_type"]["mapping"][sub_intent][0]
            intent_group = SCHEMA["intent_group"]["mapping"][issue_type][0]
            if level == "normal":
                labels.append(build_granularity_label(intent_group, issue_type))
            else:
                labels.append(build_granularity_label(intent_group, issue_type, sub_intent))
    return _dedupe_preserving_order(labels)


def _granularity_labels_for_family(family: str, level: str) -> list[str]:
    if family == "organizational":
        return _organizational_granularity_values(level)
    return _intent_granularity_values(level)


def _granularity_schema_fields() -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for family, label_fields in GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in label_fields.items():
            fields[field_name] = {"values": _granularity_labels_for_family(family, level)}
    return fields


def _insert_granularity_schema_fields(
    schema: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    updated_schema: dict[str, dict[str, Any]] = {}
    for field, config in schema.items():
        if field == "system_name":
            updated_schema.update(_granularity_schema_fields())
        updated_schema[field] = config
    return updated_schema


SCHEMA = _insert_granularity_schema_fields(SCHEMA)

ISSUE_TYPE_ALIASES = {
    "people_record_request": "people_administration_issue",
    "onboarding_offboarding": "onboarding_process_issue",
    "vendor_purchase_request": "procurement_workflow_issue",
    "workspace_support": "workplace_support_issue",
    "legal_compliance_request": "legal_review_issue",
    "sales_system_issue": "sales_platform_issue",
    "sales_operations_request": "sales_operations_issue",
    "marketing_system_issue": "marketing_platform_issue",
    "marketing_operations_request": "marketing_operations_issue",
    "customer_support_system_issue": "customer_support_platform_issue",
    "customer_support_operations_request": "customer_support_operations_issue",
    "engineering_operations_request": "release_operations_issue",
    "data_operations_request": "metrics_governance_issue",
    "travel_support_request": "travel_operations_issue",
    "workplace_experience_request": "workplace_experience_issue",
    "payroll_request": "payroll_operations_issue",
    "regional_operations_request": "regional_operations_issue",
}

SURFACE_METADATA_FIELDS = (
    "department",
    "requester_role",
    "ticket_category",
    "issue_type",
    "intent_group",
    "sub_intent",
    "system_name",
    "business_event_context",
    "writing_style",
    "clarity_level",
)

DEPARTMENT_WEIGHTS = {
    "HR": 0.75,
    "Finance": 0.85,
    "IT": 3.8,
    "Security": 1.4,
    "Facilities": 1.1,
    "Procurement": 0.6,
    "Legal": 0.45,
    "Sales": 0.95,
    "Marketing": 0.75,
    "Customer Support": 0.95,
    "Engineering": 1.6,
    "Data": 0.9,
    "Travel": 0.5,
    "Workplace Experience": 0.75,
    "Payroll": 0.65,
    "Regional Operations": 0.45,
}

CATEGORY_WEIGHTS = {
    "HR": {
        "data_update": 0.32,
        "policy_question": 0.14,
        "benefits_issue": 0.2,
        "onboarding_task": 0.34,
    },
    "Finance": {
        "reimbursement": 0.22,
        "invoice_issue": 0.33,
        "vendor_setup": 0.18,
        "purchase_request": 0.27,
    },
    "IT": {
        "access_request": 0.4,
        "account_issue": 0.24,
        "software_issue": 0.17,
        "service_request": 0.19,
    },
    "Security": {
        "security_concern": 0.34,
        "access_review": 0.4,
        "incident_report": 0.26,
    },
    "Facilities": {
        "maintenance_request": 0.3,
        "badge_issue": 0.36,
        "workspace_request": 0.34,
    },
    "Procurement": {
        "vendor_request": 0.36,
        "purchase_approval": 0.34,
        "contract_purchase": 0.3,
    },
    "Legal": {
        "contract_review": 0.44,
        "privacy_request": 0.28,
        "policy_clarification": 0.28,
    },
    "Sales": {
        "crm_support": 0.42,
        "quote_operations": 0.28,
        "territory_management": 0.30,
    },
    "Marketing": {
        "campaign_operations": 0.38,
        "marketing_tools": 0.36,
        "event_marketing": 0.26,
    },
    "Customer Support": {
        "support_platform": 0.36,
        "case_operations": 0.34,
        "customer_communications": 0.30,
    },
    "Engineering": {
        "developer_access": 0.36,
        "developer_tooling": 0.36,
        "release_operations": 0.28,
    },
    "Data": {
        "analytics_access": 0.36,
        "data_platform": 0.36,
        "metrics_governance": 0.28,
    },
    "Travel": {
        "travel_booking": 0.36,
        "travel_expense": 0.34,
        "traveler_support": 0.30,
    },
    "Workplace Experience": {
        "office_services": 0.36,
        "event_support": 0.30,
        "visitor_services": 0.34,
    },
    "Payroll": {
        "pay_cycle_issue": 0.42,
        "tax_documents": 0.24,
        "compensation_changes": 0.34,
    },
    "Regional Operations": {
        "site_operations": 0.40,
        "regional_compliance": 0.24,
        "local_services": 0.36,
    },
}

SUB_DEPARTMENT_WEIGHTS = {
    ("HR", "data_update"): {
        "Payroll": 0.34,
        "Benefits": 0.18,
        "Onboarding": 0.14,
        "Employee Relations": 0.34,
    },
    ("HR", "benefits_issue"): {
        "Payroll": 0.08,
        "Benefits": 0.72,
        "Onboarding": 0.05,
        "Employee Relations": 0.15,
    },
    ("HR", "policy_question"): {
        "Payroll": 0.1,
        "Benefits": 0.34,
        "Onboarding": 0.08,
        "Employee Relations": 0.48,
    },
    ("HR", "onboarding_task"): {
        "Payroll": 0.05,
        "Benefits": 0.1,
        "Onboarding": 0.72,
        "Employee Relations": 0.13,
    },
    ("Finance", "reimbursement"): {
        "Accounts Payable": 0.58,
        "Accounts Receivable": 0.1,
        "FP&A": 0.2,
        "Tax": 0.12,
    },
    ("Finance", "invoice_issue"): {
        "Accounts Payable": 0.62,
        "Accounts Receivable": 0.16,
        "FP&A": 0.12,
        "Tax": 0.1,
    },
    ("Finance", "vendor_setup"): {
        "Accounts Payable": 0.43,
        "Accounts Receivable": 0.08,
        "FP&A": 0.12,
        "Tax": 0.37,
    },
    ("Finance", "purchase_request"): {
        "Accounts Payable": 0.28,
        "Accounts Receivable": 0.07,
        "FP&A": 0.48,
        "Tax": 0.17,
    },
    ("IT", "access_request"): {
        "Help Desk": 0.22,
        "IAM": 0.58,
        "Network": 0.08,
        "Enterprise Apps": 0.12,
    },
    ("IT", "account_issue"): {
        "Help Desk": 0.25,
        "IAM": 0.56,
        "Network": 0.05,
        "Enterprise Apps": 0.14,
    },
    ("IT", "software_issue"): {
        "Help Desk": 0.34,
        "IAM": 0.08,
        "Network": 0.1,
        "Enterprise Apps": 0.48,
    },
    ("IT", "service_request"): {
        "Help Desk": 0.58,
        "IAM": 0.05,
        "Network": 0.14,
        "Enterprise Apps": 0.23,
    },
    ("Security", "security_concern"): {
        "SecOps": 0.46,
        "GRC": 0.08,
        "Incident Response": 0.34,
        "Identity Governance": 0.12,
    },
    ("Security", "access_review"): {
        "SecOps": 0.1,
        "GRC": 0.28,
        "Incident Response": 0.07,
        "Identity Governance": 0.55,
    },
    ("Security", "incident_report"): {
        "SecOps": 0.34,
        "GRC": 0.05,
        "Incident Response": 0.56,
        "Identity Governance": 0.05,
    },
    ("Facilities", "maintenance_request"): {
        "Maintenance": 0.7,
        "Badge Access": 0.06,
        "Workspace": 0.09,
        "Office Services": 0.15,
    },
    ("Facilities", "badge_issue"): {
        "Maintenance": 0.04,
        "Badge Access": 0.74,
        "Workspace": 0.07,
        "Office Services": 0.15,
    },
    ("Facilities", "workspace_request"): {
        "Maintenance": 0.1,
        "Badge Access": 0.05,
        "Workspace": 0.58,
        "Office Services": 0.27,
    },
    ("Procurement", "vendor_request"): {
        "Vendor Management": 0.7,
        "Purchase Requests": 0.17,
        "Asset Procurement": 0.13,
    },
    ("Procurement", "purchase_approval"): {
        "Vendor Management": 0.1,
        "Purchase Requests": 0.58,
        "Asset Procurement": 0.32,
    },
    ("Procurement", "contract_purchase"): {
        "Vendor Management": 0.35,
        "Purchase Requests": 0.18,
        "Asset Procurement": 0.47,
    },
    ("Legal", "contract_review"): {
        "Contract Review": 0.7,
        "Privacy": 0.07,
        "Compliance Support": 0.23,
    },
    ("Legal", "privacy_request"): {
        "Contract Review": 0.08,
        "Privacy": 0.74,
        "Compliance Support": 0.18,
    },
    ("Legal", "policy_clarification"): {
        "Contract Review": 0.18,
        "Privacy": 0.16,
        "Compliance Support": 0.66,
    },
    ("Sales", "crm_support"): {
        "Sales Operations": 0.46,
        "Deal Desk": 0.18,
        "Territory Planning": 0.36,
    },
    ("Sales", "quote_operations"): {
        "Sales Operations": 0.18,
        "Deal Desk": 0.64,
        "Territory Planning": 0.18,
    },
    ("Sales", "territory_management"): {
        "Sales Operations": 0.18,
        "Deal Desk": 0.12,
        "Territory Planning": 0.70,
    },
    ("Marketing", "campaign_operations"): {
        "Marketing Operations": 0.56,
        "Demand Generation": 0.30,
        "Events": 0.14,
    },
    ("Marketing", "marketing_tools"): {
        "Marketing Operations": 0.46,
        "Demand Generation": 0.28,
        "Events": 0.26,
    },
    ("Marketing", "event_marketing"): {
        "Marketing Operations": 0.16,
        "Demand Generation": 0.14,
        "Events": 0.70,
    },
    ("Customer Support", "support_platform"): {
        "Support Operations": 0.52,
        "Escalations": 0.14,
        "Knowledge Management": 0.34,
    },
    ("Customer Support", "case_operations"): {
        "Support Operations": 0.56,
        "Escalations": 0.30,
        "Knowledge Management": 0.14,
    },
    ("Customer Support", "customer_communications"): {
        "Support Operations": 0.30,
        "Escalations": 0.12,
        "Knowledge Management": 0.58,
    },
    ("Engineering", "developer_access"): {
        "Developer Productivity": 0.56,
        "Infrastructure": 0.30,
        "Release Engineering": 0.14,
    },
    ("Engineering", "developer_tooling"): {
        "Developer Productivity": 0.42,
        "Infrastructure": 0.42,
        "Release Engineering": 0.16,
    },
    ("Engineering", "release_operations"): {
        "Developer Productivity": 0.16,
        "Infrastructure": 0.18,
        "Release Engineering": 0.66,
    },
    ("Data", "analytics_access"): {
        "Analytics Engineering": 0.26,
        "Business Intelligence": 0.58,
        "Data Governance": 0.16,
    },
    ("Data", "data_platform"): {
        "Analytics Engineering": 0.52,
        "Business Intelligence": 0.14,
        "Data Governance": 0.34,
    },
    ("Data", "metrics_governance"): {
        "Analytics Engineering": 0.18,
        "Business Intelligence": 0.26,
        "Data Governance": 0.56,
    },
    ("Travel", "travel_booking"): {
        "Travel Desk": 0.64,
        "Expense Operations": 0.18,
        "Mobility": 0.18,
    },
    ("Travel", "travel_expense"): {
        "Travel Desk": 0.18,
        "Expense Operations": 0.66,
        "Mobility": 0.16,
    },
    ("Travel", "traveler_support"): {
        "Travel Desk": 0.26,
        "Expense Operations": 0.10,
        "Mobility": 0.64,
    },
    ("Workplace Experience", "office_services"): {
        "Office Services": 0.62,
        "Events and Hospitality": 0.20,
        "Visitor Services": 0.18,
    },
    ("Workplace Experience", "event_support"): {
        "Office Services": 0.14,
        "Events and Hospitality": 0.72,
        "Visitor Services": 0.14,
    },
    ("Workplace Experience", "visitor_services"): {
        "Office Services": 0.16,
        "Events and Hospitality": 0.14,
        "Visitor Services": 0.70,
    },
    ("Payroll", "pay_cycle_issue"): {
        "Payroll Operations": 0.68,
        "Tax Operations": 0.12,
        "Compensation Administration": 0.20,
    },
    ("Payroll", "tax_documents"): {
        "Payroll Operations": 0.18,
        "Tax Operations": 0.68,
        "Compensation Administration": 0.14,
    },
    ("Payroll", "compensation_changes"): {
        "Payroll Operations": 0.22,
        "Tax Operations": 0.08,
        "Compensation Administration": 0.70,
    },
    ("Regional Operations", "site_operations"): {
        "Site Operations": 0.66,
        "Local Compliance": 0.12,
        "Regional Services": 0.22,
    },
    ("Regional Operations", "regional_compliance"): {
        "Site Operations": 0.12,
        "Local Compliance": 0.70,
        "Regional Services": 0.18,
    },
    ("Regional Operations", "local_services"): {
        "Site Operations": 0.20,
        "Local Compliance": 0.18,
        "Regional Services": 0.62,
    },
}

BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS = {
    "routine_operation": 0.42,
    "new_hire_onboarding": 0.09,
    "employee_offboarding": 0.07,
    "quarter_end_close": 0.08,
    "audit_preparation": 0.09,
    "org_restructure": 0.09,
    "product_launch": 0.09,
    "office_move": 0.07,
}

BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS = {
    "new_hire_onboarding": {
        "HR",
        "IT",
        "Facilities",
        "Engineering",
        "Data",
        "Payroll",
        "Workplace Experience",
    },
    "office_move": {
        "Facilities",
        "IT",
        "Workplace Experience",
        "Regional Operations",
    },
    "quarter_end_close": {
        "Finance",
        "Procurement",
        "Payroll",
        "Travel",
    },
    "audit_preparation": {
        "Security",
        "Finance",
        "Legal",
        "Data",
        "Payroll",
        "Regional Operations",
    },
}

BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS = {
    "new_hire_onboarding": {
        "HR": 0.30,
        "IT": 0.26,
        "Engineering": 0.16,
        "Payroll": 0.12,
        "Data": 0.08,
        "Workplace Experience": 0.08,
    },
    "office_move": {
        "Facilities": 0.44,
        "IT": 0.16,
        "Workplace Experience": 0.24,
        "Regional Operations": 0.16,
    },
}

BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS = {
    "HR": {
        "new_hire_onboarding": 0.22,
        "org_restructure": 0.10,
        "employee_offboarding": 0.06,
    },
    "Finance": {
        "quarter_end_close": 0.22,
        "audit_preparation": 0.08,
    },
    "IT": {
        "new_hire_onboarding": 0.12,
        "product_launch": 0.10,
        "office_move": 0.08,
    },
    "Security": {
        "audit_preparation": 0.14,
        "employee_offboarding": 0.10,
        "product_launch": 0.06,
    },
    "Facilities": {
        "office_move": 0.28,
        "new_hire_onboarding": 0.05,
    },
    "Procurement": {
        "product_launch": 0.14,
        "quarter_end_close": 0.08,
    },
    "Legal": {
        "audit_preparation": 0.12,
        "product_launch": 0.08,
        "org_restructure": 0.05,
    },
    "Sales": {
        "product_launch": 0.18,
        "org_restructure": 0.10,
        "quarter_end_close": 0.04,
    },
    "Marketing": {
        "product_launch": 0.20,
        "org_restructure": 0.08,
    },
    "Customer Support": {
        "product_launch": 0.10,
        "org_restructure": 0.06,
    },
    "Engineering": {
        "product_launch": 0.18,
        "new_hire_onboarding": 0.08,
        "org_restructure": 0.06,
    },
    "Data": {
        "audit_preparation": 0.10,
        "product_launch": 0.08,
        "new_hire_onboarding": 0.06,
    },
    "Travel": {
        "quarter_end_close": 0.10,
        "office_move": 0.08,
    },
    "Workplace Experience": {
        "office_move": 0.22,
        "product_launch": 0.08,
        "new_hire_onboarding": 0.06,
    },
    "Payroll": {
        "quarter_end_close": 0.18,
        "new_hire_onboarding": 0.08,
        "employee_offboarding": 0.06,
    },
    "Regional Operations": {
        "office_move": 0.18,
        "org_restructure": 0.10,
        "audit_preparation": 0.06,
    },
}

BUSINESS_EVENT_CONTEXT_CATEGORY_ADJUSTMENTS = {
    "onboarding_task": {
        "new_hire_onboarding": 0.40,
    },
    "access_review": {
        "audit_preparation": 0.14,
        "employee_offboarding": 0.10,
    },
    "workspace_request": {
        "office_move": 0.24,
    },
    "purchase_approval": {
        "product_launch": 0.12,
        "quarter_end_close": 0.08,
    },
    "contract_review": {
        "product_launch": 0.12,
        "audit_preparation": 0.05,
    },
    "invoice_issue": {
        "quarter_end_close": 0.10,
    },
    "developer_access": {
        "new_hire_onboarding": 0.20,
        "employee_offboarding": 0.08,
    },
    "analytics_access": {
        "new_hire_onboarding": 0.16,
    },
    "release_operations": {
        "product_launch": 0.18,
    },
    "campaign_operations": {
        "product_launch": 0.16,
    },
    "office_services": {
        "office_move": 0.12,
    },
    "site_operations": {
        "office_move": 0.14,
        "org_restructure": 0.06,
    },
    "regional_compliance": {
        "audit_preparation": 0.14,
    },
    "travel_expense": {
        "quarter_end_close": 0.08,
    },
}

BUSINESS_EVENT_CONTEXT_SUB_INTENT_ADJUSTMENTS = {
    "new_hire_setup": {
        "new_hire_onboarding": 0.36,
    },
    "orientation_question": {
        "new_hire_onboarding": 0.36,
    },
    "missing_onboarding_step": {
        "new_hire_onboarding": 0.36,
    },
    "terminated_user_access_check": {
        "employee_offboarding": 0.14,
        "audit_preparation": 0.12,
    },
    "role_audit": {
        "employee_offboarding": 0.14,
        "audit_preparation": 0.12,
    },
    "malware_alert": {
        "routine_operation": 0.08,
    },
    "credential_compromise": {
        "routine_operation": 0.08,
    },
    "phishing_report": {
        "routine_operation": 0.08,
    },
    "desk_move": {
        "office_move": 0.14,
    },
    "meeting_room_setup": {
        "office_move": 0.14,
    },
    "ergonomic_equipment": {
        "office_move": 0.14,
    },
    "repo_access_request": {
        "new_hire_onboarding": 0.14,
    },
    "cloud_role_access": {
        "new_hire_onboarding": 0.14,
    },
    "dashboard_access_request": {
        "new_hire_onboarding": 0.14,
    },
    "warehouse_role_access": {
        "new_hire_onboarding": 0.14,
    },
    "campaign_launch_change": {
        "product_launch": 0.14,
    },
    "campaign_schedule_update": {
        "product_launch": 0.12,
    },
    "asset_approval_delay": {
        "product_launch": 0.12,
    },
    "release_window_question": {
        "product_launch": 0.14,
    },
    "deployment_freeze_exception": {
        "product_launch": 0.14,
    },
    "status_page_update_request": {
        "product_launch": 0.14,
    },
    "all_hands_room_reset": {
        "office_move": 0.12,
    },
    "parking_access_request": {
        "office_move": 0.10,
    },
    "visitor_list_sync_issue": {
        "office_move": 0.08,
    },
    "local_site_access_issue": {
        "office_move": 0.12,
    },
    "regional_move_coordination": {
        "office_move": 0.12,
    },
    "site_readiness_check": {
        "office_move": 0.12,
    },
    "regional_seating_change": {
        "office_move": 0.10,
    },
    "regional_audit_evidence_request": {
        "audit_preparation": 0.12,
    },
    "build_platform_access": {
        "new_hire_onboarding": 0.14,
    },
    "service_account_access": {
        "new_hire_onboarding": 0.12,
    },
    "bi_group_membership_change": {
        "new_hire_onboarding": 0.12,
    },
    "row_level_access_issue": {
        "new_hire_onboarding": 0.12,
        "audit_preparation": 0.08,
    },
}

TICKET_CATEGORY_EVENT_CONTEXT_ADJUSTMENTS = {
    "new_hire_onboarding": {
        "HR": {
            "onboarding_task": 0.55,
            "benefits_issue": 0.12,
        },
        "IT": {
            "access_request": 0.25,
            "service_request": 0.20,
        },
        "Facilities": {
            "badge_issue": 0.10,
        },
        "Engineering": {
            "developer_access": 0.20,
        },
        "Data": {
            "analytics_access": 0.10,
        },
        "Payroll": {
            "pay_cycle_issue": 0.14,
            "compensation_changes": 0.10,
        },
        "Workplace Experience": {
            "visitor_services": 0.10,
        },
    },
    "employee_offboarding": {
        "Security": {
            "access_review": 0.35,
        },
        "IT": {
            "access_request": 0.16,
            "account_issue": 0.12,
        },
        "Legal": {
            "policy_clarification": 0.08,
        },
    },
    "quarter_end_close": {
        "Finance": {
            "invoice_issue": 0.20,
            "purchase_request": 0.14,
        },
        "Procurement": {
            "purchase_approval": 0.10,
        },
        "Payroll": {
            "pay_cycle_issue": 0.18,
            "compensation_changes": 0.08,
        },
        "Travel": {
            "travel_expense": 0.10,
        },
    },
    "audit_preparation": {
        "Security": {
            "access_review": 0.24,
        },
        "Finance": {
            "vendor_setup": 0.12,
        },
        "Legal": {
            "privacy_request": 0.14,
            "policy_clarification": 0.10,
        },
        "Data": {
            "metrics_governance": 0.14,
        },
        "Payroll": {
            "tax_documents": 0.12,
        },
        "Regional Operations": {
            "regional_compliance": 0.10,
        },
    },
    "product_launch": {
        "IT": {
            "access_request": 0.15,
            "software_issue": 0.08,
        },
        "Procurement": {
            "purchase_approval": 0.18,
            "contract_purchase": 0.10,
        },
        "Legal": {
            "contract_review": 0.15,
        },
        "Sales": {
            "crm_support": 0.12,
            "quote_operations": 0.10,
        },
        "Marketing": {
            "campaign_operations": 0.18,
            "marketing_tools": 0.08,
        },
        "Customer Support": {
            "customer_communications": 0.14,
            "support_platform": 0.08,
        },
        "Engineering": {
            "release_operations": 0.20,
            "developer_access": 0.08,
        },
        "Data": {
            "data_platform": 0.10,
            "analytics_access": 0.08,
        },
    },
    "office_move": {
        "Facilities": {
            "workspace_request": 0.30,
            "maintenance_request": 0.12,
        },
        "IT": {
            "service_request": 0.15,
        },
        "Workplace Experience": {
            "office_services": 0.18,
            "event_support": 0.10,
        },
        "Regional Operations": {
            "site_operations": 0.18,
            "local_services": 0.10,
        },
    },
}

REQUESTER_ROLE_EVENT_CONTEXT_ADJUSTMENTS = {
    "new_hire_onboarding": {
        "new_hire": 0.25,
        "manager": 0.14,
        "executive_assistant": 0.06,
        "contractor": -0.03,
    },
    "employee_offboarding": {
        "manager": 0.18,
        "director": 0.08,
        "executive_assistant": 0.05,
        "new_hire": -0.09,
        "intern": -0.03,
    },
    "quarter_end_close": {
        "employee": 0.08,
        "manager": 0.05,
        "director": 0.05,
    },
    "product_launch": {
        "manager": 0.06,
        "director": 0.08,
        "executive_assistant": 0.05,
    },
}

URGENCY_EVENT_CONTEXT_ADJUSTMENTS = {
    "quarter_end_close": {
        "high": 0.18,
    },
    "new_hire_onboarding": {
        "high": 0.10,
    },
    "product_launch": {
        "high": 0.08,
    },
    "office_move": {
        "medium": 0.08,
    },
}

OWNING_TEAM_BASE_WEIGHTS = {
    "HR": {
        "HR Operations": 0.56,
        "Benefits Administration": 0.20,
        "People Services": 0.24,
    },
    "Finance": {
        "Finance Operations": 0.50,
        "Accounts Payable Operations": 0.28,
        "Procurement Finance": 0.22,
    },
    "IT": {
        "IT Support": 0.56,
        "IAM Team": 0.44,
    },
    "Security": {
        "Security Team": 0.52,
        "Identity Security": 0.24,
        "Incident Response": 0.24,
    },
    "Facilities": {
        "Facilities Team": 0.54,
        "Badge Services": 0.20,
        "Workplace Services": 0.26,
    },
    "Procurement": {
        "Procurement Team": 0.54,
        "Vendor Operations": 0.24,
        "Purchasing Desk": 0.22,
    },
    "Legal": {
        "Legal Ops": 0.50,
        "Privacy Review": 0.20,
        "Commercial Legal": 0.30,
    },
    "Sales": {
        "Sales Operations": 0.56,
        "Revenue Systems": 0.44,
    },
    "Marketing": {
        "Marketing Operations": 0.60,
        "MarTech Support": 0.40,
    },
    "Customer Support": {
        "Support Operations": 0.62,
        "Customer Systems": 0.38,
    },
    "Engineering": {
        "Developer Productivity": 0.58,
        "Release Engineering": 0.42,
    },
    "Data": {
        "Data Platform": 0.54,
        "Analytics Operations": 0.46,
    },
    "Travel": {
        "Travel Operations": 0.66,
        "Expense Audit": 0.34,
    },
    "Workplace Experience": {
        "Workplace Experience Team": 0.56,
        "Events Support": 0.24,
        "Visitor Services": 0.20,
    },
    "Payroll": {
        "Payroll Operations": 0.72,
        "Payroll Tax Support": 0.28,
    },
    "Regional Operations": {
        "Regional Operations Team": 0.56,
        "Site Services": 0.22,
        "Regional Compliance": 0.22,
    },
}

OWNING_TEAM_CATEGORY_ADJUSTMENTS = {
    "HR": {
        "data_update": {
            "HR Operations": 0.18,
        },
        "benefits_issue": {
            "Benefits Administration": 0.34,
        },
        "onboarding_task": {
            "People Services": 0.30,
        },
    },
    "Finance": {
        "reimbursement": {
            "Accounts Payable Operations": 0.20,
        },
        "invoice_issue": {
            "Accounts Payable Operations": 0.28,
        },
        "vendor_setup": {
            "Finance Operations": 0.10,
            "Accounts Payable Operations": 0.10,
        },
        "purchase_request": {
            "Procurement Finance": 0.22,
        },
    },
    "IT": {
        "access_request": {
            "IAM Team": 0.22,
        },
        "account_issue": {
            "IAM Team": 0.16,
        },
        "software_issue": {
            "IT Support": 0.18,
        },
        "service_request": {
            "IT Support": 0.18,
        },
    },
    "Security": {
        "security_concern": {
            "Security Team": 0.14,
        },
        "access_review": {
            "Identity Security": 0.30,
        },
        "incident_report": {
            "Incident Response": 0.30,
        },
    },
    "Facilities": {
        "maintenance_request": {
            "Facilities Team": 0.24,
        },
        "badge_issue": {
            "Badge Services": 0.34,
        },
        "workspace_request": {
            "Workplace Services": 0.28,
        },
    },
    "Procurement": {
        "vendor_request": {
            "Vendor Operations": 0.32,
        },
        "purchase_approval": {
            "Purchasing Desk": 0.30,
        },
        "contract_purchase": {
            "Procurement Team": 0.18,
        },
    },
    "Legal": {
        "contract_review": {
            "Commercial Legal": 0.34,
        },
        "privacy_request": {
            "Privacy Review": 0.36,
        },
        "policy_clarification": {
            "Legal Ops": 0.20,
        },
    },
    "Sales": {
        "crm_support": {
            "Revenue Systems": 0.20,
        },
        "quote_operations": {
            "Revenue Systems": 0.20,
        },
        "territory_management": {
            "Sales Operations": 0.22,
        },
    },
    "Marketing": {
        "marketing_tools": {
            "MarTech Support": 0.22,
        },
        "campaign_operations": {
            "Marketing Operations": 0.12,
        },
        "event_marketing": {
            "Marketing Operations": 0.12,
        },
    },
    "Customer Support": {
        "support_platform": {
            "Customer Systems": 0.24,
        },
        "case_operations": {
            "Support Operations": 0.18,
        },
    },
    "Engineering": {
        "release_operations": {
            "Release Engineering": 0.24,
        },
        "developer_access": {
            "Developer Productivity": 0.18,
        },
        "developer_tooling": {
            "Developer Productivity": 0.18,
        },
    },
    "Data": {
        "data_platform": {
            "Data Platform": 0.24,
        },
        "analytics_access": {
            "Analytics Operations": 0.12,
        },
        "metrics_governance": {
            "Analytics Operations": 0.12,
        },
    },
    "Travel": {
        "travel_booking": {
            "Travel Operations": 0.18,
        },
        "travel_expense": {
            "Expense Audit": 0.28,
        },
    },
    "Workplace Experience": {
        "office_services": {
            "Workplace Experience Team": 0.20,
        },
        "event_support": {
            "Events Support": 0.34,
        },
        "visitor_services": {
            "Visitor Services": 0.36,
        },
    },
    "Payroll": {
        "pay_cycle_issue": {
            "Payroll Operations": 0.22,
        },
        "tax_documents": {
            "Payroll Tax Support": 0.36,
        },
    },
    "Regional Operations": {
        "site_operations": {
            "Site Services": 0.34,
        },
        "regional_compliance": {
            "Regional Compliance": 0.36,
        },
        "local_services": {
            "Regional Operations Team": 0.18,
        },
    },
}

ACCESS_ISSUES = (
    "account_locked",
    "admin_access",
    "agent_provisioning_request",
    "badge_not_working",
    "bi_group_membership_change",
    "build_platform_access",
    "cloud_role_access",
    "crm_access_change",
    "dashboard_access_request",
    "feature_flag_access",
    "marketing_tool_access",
    "marketing_permission_change",
    "mfa_problem",
    "new_badge_request",
    "parking_access_request",
    "queue_access_change",
    "password_reset",
    "privileged_access_review",
    "repo_access_request",
    "row_level_access_issue",
    "role_audit",
    "service_account_access",
    "shared_drive_access",
    "temporary_access",
    "terminated_user_access_check",
    "visitor_access",
    "vpn_access",
    "warehouse_role_access",
    "after_hours_access_setup",
    "local_site_access_issue",
)

PRIVILEGED_ACCESS_ISSUES = (
    "admin_access",
    "cloud_role_access",
    "feature_flag_access",
    "privileged_access_review",
    "role_audit",
    "service_account_access",
    "temporary_access",
    "terminated_user_access_check",
    "warehouse_role_access",
)

SECURITY_ISSUES = (
    "credential_compromise",
    "data_exposure_concern",
    "malware_alert",
    "phishing_report",
    "policy_violation",
    "privileged_access_review",
    "role_audit",
    "suspicious_login",
    "terminated_user_access_check",
)

CRITICAL_SECURITY_ISSUES = (
    "credential_compromise",
    "data_exposure_concern",
    "malware_alert",
    "phishing_report",
    "suspicious_login",
)

PURCHASE_ISSUES = (
    "approval_missing",
    "budget_code_question",
    "equipment_purchase",
    "po_creation",
    "quote_review",
    "service_contract_request",
    "software_purchase",
    "subscription_renewal",
    "urgent_purchase_exception",
)

VENDOR_ISSUES = (
    "bank_detail_update",
    "catering_setup_issue",
    "local_vendor_followup",
    "missing_tax_form",
    "new_supplier_request",
    "new_vendor_creation",
    "quote_review",
    "service_contract_request",
    "swag_vendor_followup",
    "vendor_document_missing",
    "vendor_update",
)

DOCUMENT_ISSUES = (
    "data_access_request",
    "data_deletion_request",
    "missing_tax_form",
    "msa_review",
    "nda_review",
    "pay_stub_request",
    "privacy_clause_question",
    "quote_review",
    "regional_audit_evidence_request",
    "receipt_followup",
    "tax_document_delivery_issue",
    "vendor_contract_review",
    "vendor_document_missing",
    "visa_letter_request",
    "w2_correction_request",
)

SYSTEM_SYNC_ISSUES = (
    "account_locked",
    "application_crash",
    "audience_sync_problem",
    "badge_not_working",
    "hardware_replacement",
    "installation_failure",
    "laptop_setup",
    "lead_routing_issue",
    "license_missing",
    "manager_change",
    "mfa_problem",
    "nurture_enrollment_problem",
    "password_reset",
    "personal_info_correction",
    "quote_builder_sync_issue",
    "scheduled_report_missing",
    "software_installation",
    "stale_dataset_alert",
    "tracking_gap_report",
    "traveler_profile_sync_issue",
    "title_change",
    "visitor_list_sync_issue",
)

MAIN_POV_CROSSTABS = (
    ("department", "ticket_category"),
    ("intent_group", "issue_type"),
    ("issue_type", "sub_intent"),
    ("department", "system_name"),
    ("urgency", "severity"),
)

REQUESTER_ROLE_BASE_WEIGHTS = {
    "employee": 0.40,
    "manager": 0.20,
    "director": 0.10,
    "contractor": 0.08,
    "intern": 0.05,
    "executive_assistant": 0.07,
    "new_hire": 0.10,
}

REQUESTER_ROLE_DEPARTMENT_ADJUSTMENTS = {
    "Legal": {
        "manager": 0.05,
        "director": 0.04,
        "executive_assistant": 0.03,
    },
    "Procurement": {
        "manager": 0.05,
        "director": 0.04,
        "executive_assistant": 0.03,
    },
    "Facilities": {
        "employee": 0.06,
        "intern": 0.02,
    },
    "Security": {
        "manager": 0.03,
        "director": 0.03,
    },
    "Sales": {
        "manager": 0.05,
        "director": 0.05,
        "executive_assistant": 0.02,
    },
    "Marketing": {
        "manager": 0.05,
        "director": 0.05,
        "executive_assistant": 0.02,
    },
    "Customer Support": {
        "employee": 0.08,
        "manager": 0.03,
    },
    "Engineering": {
        "employee": 0.06,
        "contractor": 0.04,
    },
    "Data": {
        "employee": 0.06,
        "contractor": 0.04,
    },
    "Travel": {
        "executive_assistant": 0.10,
        "manager": 0.04,
    },
    "Workplace Experience": {
        "employee": 0.06,
        "manager": 0.04,
    },
    "Payroll": {
        "employee": 0.06,
        "manager": 0.04,
    },
    "Regional Operations": {
        "employee": 0.06,
        "manager": 0.04,
    },
}

REQUESTER_ROLE_CATEGORY_ADJUSTMENTS = {
    "contract_review": {
        "manager": 0.08,
        "director": 0.08,
        "executive_assistant": 0.05,
    },
    "purchase_approval": {
        "manager": 0.08,
        "director": 0.08,
        "executive_assistant": 0.05,
    },
    "contract_purchase": {
        "manager": 0.08,
        "director": 0.08,
        "executive_assistant": 0.05,
    },
    "travel_booking": {
        "executive_assistant": 0.10,
    },
    "event_marketing": {
        "executive_assistant": 0.10,
    },
}

SYSTEM_NAME_BASE_ADJUSTMENTS = {
    "HR": {
        "Workday": 1.2,
    },
}

SYSTEM_NAME_CATEGORY_ADJUSTMENTS = {
    "HR": {
        "onboarding_task": {
            "Greenhouse": 0.7,
        },
    },
    "Finance": {
        "reimbursement": {
            "Expense Portal": 1.3,
        },
        "invoice_issue": {
            "NetSuite": 1.1,
            "SAP": 0.4,
        },
        "vendor_setup": {
            "SAP": 0.8,
            "Coupa": 0.6,
        },
        "purchase_request": {
            "Coupa": 1.0,
            "SAP": 0.6,
        },
    },
    "IT": {
        "software_issue": {
            "Jira": 0.8,
            "Microsoft 365": 0.4,
            "Google Workspace": 0.4,
        },
        "service_request": {
            "Jira": 1.0,
        },
    },
    "Security": {
        "access_review": {
            "Access Governance Tool": 1.4,
            "Okta": 0.4,
        },
    },
    "Facilities": {
        "maintenance_request": {
            "Workplace Portal": 0.8,
        },
        "badge_issue": {
            "Badge System": 1.8,
        },
        "workspace_request": {
            "Workplace Portal": 0.8,
        },
    },
    "Procurement": {
        "vendor_request": {
            "Vendor Portal": 1.4,
            "SAP": 0.4,
        },
        "purchase_approval": {
            "Coupa": 1.4,
        },
        "contract_purchase": {
            "SAP": 0.8,
            "Coupa": 0.8,
        },
    },
    "Legal": {
        "contract_review": {
            "Contract Lifecycle Tool": 1.7,
        },
        "privacy_request": {
            "Document Management System": 1.1,
            "Contract Lifecycle Tool": 0.4,
        },
        "policy_clarification": {
            "Document Management System": 1.1,
            "Contract Lifecycle Tool": 0.4,
        },
    },
    "Sales": {
        "crm_support": {
            "Salesforce": 1.5,
            "Sales Engagement Platform": 0.4,
        },
        "quote_operations": {
            "Quote Builder": 1.6,
            "Salesforce": 0.4,
        },
        "territory_management": {
            "Salesforce": 1.1,
        },
    },
    "Marketing": {
        "campaign_operations": {
            "Marketo": 1.3,
            "HubSpot": 0.4,
        },
        "marketing_tools": {
            "Marketo": 1.0,
            "HubSpot": 1.0,
            "CMS Platform": 0.4,
        },
        "event_marketing": {
            "HubSpot": 0.8,
            "CMS Platform": 0.6,
        },
    },
    "Customer Support": {
        "support_platform": {
            "Zendesk": 1.6,
        },
        "case_operations": {
            "Zendesk": 1.2,
            "Help Center": 0.3,
        },
        "customer_communications": {
            "Statuspage": 1.2,
            "Help Center": 1.0,
        },
    },
    "Engineering": {
        "developer_tooling": {
            "Build Platform": 1.5,
            "GitHub": 0.3,
        },
        "release_operations": {
            "Feature Flag Service": 1.1,
            "Build Platform": 0.8,
        },
    },
    "Data": {
        "analytics_access": {
            "Looker": 1.4,
            "Snowflake": 0.6,
        },
        "data_platform": {
            "dbt Cloud": 1.2,
            "Snowflake": 0.8,
        },
        "metrics_governance": {
            "Tracking Console": 1.1,
            "Looker": 0.4,
        },
    },
    "Travel": {
        "travel_booking": {
            "Travel Portal": 1.5,
        },
        "travel_expense": {
            "Concur": 1.4,
            "Corporate Card Portal": 0.8,
        },
        "traveler_support": {
            "Travel Portal": 0.8,
        },
    },
    "Workplace Experience": {
        "office_services": {
            "Workplace Portal": 0.8,
            "Catering Portal": 1.1,
        },
        "event_support": {
            "Event Calendar": 1.3,
            "Workplace Portal": 0.4,
        },
        "visitor_services": {
            "Visitor Management System": 1.5,
            "Workplace Portal": 0.4,
        },
    },
    "Payroll": {
        "pay_cycle_issue": {
            "Payroll Portal": 1.4,
        },
        "tax_documents": {
            "ADP": 1.2,
            "Workday": 0.4,
        },
        "compensation_changes": {
            "Workday": 0.8,
            "Payroll Portal": 0.6,
        },
    },
    "Regional Operations": {
        "site_operations": {
            "Local Operations Portal": 1.0,
            "Badge System": 1.0,
        },
        "regional_compliance": {
            "Local Operations Portal": 1.2,
        },
        "local_services": {
            "Local Operations Portal": 1.1,
            "Workplace Portal": 0.4,
        },
    },
}

SYSTEM_NAME_SUB_INTENT_ADJUSTMENTS = {
    "vpn_access": {
        "VPN": 2.0,
        "Okta": 0.5,
    },
    "shared_drive_access": {
        "Google Workspace": 1.1,
        "Microsoft 365": 1.1,
    },
    "admin_access": {
        "Okta": 1.5,
    },
    "temporary_access": {
        "Okta": 1.5,
    },
    "password_reset": {
        "Okta": 1.5,
    },
    "account_locked": {
        "Okta": 1.5,
    },
    "mfa_problem": {
        "Okta": 1.4,
        "Microsoft 365": 0.4,
    },
    "phishing_report": {
        "Okta": 0.8,
        "SIEM": 1.0,
    },
    "suspicious_login": {
        "Okta": 0.8,
        "SIEM": 1.0,
    },
    "credential_compromise": {
        "Okta": 0.8,
        "SIEM": 1.0,
    },
    "data_exposure_concern": {
        "SIEM": 1.0,
        "EDR Platform": 1.0,
    },
    "malware_alert": {
        "SIEM": 1.0,
        "EDR Platform": 1.0,
    },
    "meeting_room_setup": {
        "Room Booking Tool": 1.7,
    },
    "repo_access_request": {
        "GitHub": 1.6,
    },
    "cloud_role_access": {
        "Cloud Console": 1.6,
    },
}

APPROVAL_REQUIRED_BASE_WEIGHTS = {
    "none": 0.44,
    "manager_approval": 0.24,
    "finance_approval": 0.12,
    "security_approval": 0.10,
    "multi_step_approval": 0.10,
}

APPROVAL_REQUIRED_ACCESS_SUB_INTENT_ADJUSTMENTS = {
    sub_intent: {
        "none": -0.16,
        "manager_approval": 0.14,
        "security_approval": 0.08,
    }
    for sub_intent in ACCESS_ISSUES
}

APPROVAL_REQUIRED_PRIVILEGED_ACCESS_SUB_INTENT_ADJUSTMENTS = {
    sub_intent: {
        "security_approval": 0.18,
        "multi_step_approval": 0.18,
    }
    for sub_intent in PRIVILEGED_ACCESS_ISSUES
}

APPROVAL_REQUIRED_TICKET_CATEGORY_OVERRIDES = {
    ticket_category: {"none": 0.05}
    for ticket_category in ("purchase_request", "purchase_approval", "contract_purchase")
}

APPROVAL_REQUIRED_TICKET_CATEGORY_ADJUSTMENTS = {
    **{
        ticket_category: {
            "manager_approval": 0.12,
            "finance_approval": 0.22,
            "multi_step_approval": 0.20,
        }
        for ticket_category in ("purchase_request", "purchase_approval", "contract_purchase")
    },
    **{
        ticket_category: {
            "manager_approval": 0.14,
            "multi_step_approval": 0.10,
        }
        for ticket_category in (
            "quote_operations",
            "territory_management",
            "event_marketing",
            "release_operations",
            "regional_compliance",
        )
    },
    **{
        ticket_category: {
            "none": -0.10,
            "security_approval": 0.14,
        }
        for ticket_category in ("developer_access", "analytics_access")
    },
    "vendor_setup": {
        "finance_approval": 0.16,
        "multi_step_approval": 0.10,
    },
}

APPROVAL_REQUIRED_DEPARTMENT_ADJUSTMENTS = {
    "Security": {"security_approval": 0.12},
    "Legal": {
        "none": 0.04,
        "multi_step_approval": 0.08,
    },
    "Travel": {"finance_approval": 0.08},
    "Engineering": {"security_approval": 0.06},
    "Data": {"security_approval": 0.06},
}

DEPENDENCY_TYPE_BASE_WEIGHTS = {
    "none": 0.58,
    "waiting_for_document": 0.12,
    "waiting_for_manager": 0.11,
    "waiting_for_vendor": 0.07,
    "waiting_for_system_sync": 0.08,
    "waiting_for_another_team": 0.04,
}

DEPENDENCY_TYPE_APPROVAL_REQUIRED_ADJUSTMENTS = {
    "manager_approval": {"waiting_for_manager": 0.16},
    "multi_step_approval": {"waiting_for_manager": 0.16},
}

DEPENDENCY_TYPE_VENDOR_SUB_INTENT_ADJUSTMENTS = {
    sub_intent: {
        "waiting_for_vendor": 0.22,
        "waiting_for_document": 0.10,
    }
    for sub_intent in VENDOR_ISSUES
}

DEPENDENCY_TYPE_DOCUMENT_SUB_INTENT_ADJUSTMENTS = {
    sub_intent: {"waiting_for_document": 0.18} for sub_intent in DOCUMENT_ISSUES
}

DEPENDENCY_TYPE_SYSTEM_SYNC_SUB_INTENT_ADJUSTMENTS = {
    sub_intent: {"waiting_for_system_sync": 0.18} for sub_intent in SYSTEM_SYNC_ISSUES
}

DEPENDENCY_TYPE_WORKFLOW_STAGE_OVERRIDES = {
    "blocked": {"none": 0.0},
}

DEPENDENCY_TYPE_WORKFLOW_STAGE_ADJUSTMENTS = {
    "blocked": {
        "waiting_for_another_team": 0.18,
        "waiting_for_system_sync": 0.08,
    },
    "pending_approval": {"waiting_for_manager": 0.16},
}

SEVERITY_BASE_WEIGHTS_BY_URGENCY = {
    "low": {
        "minor": 0.55,
        "moderate": 0.36,
        "major": 0.08,
        "business_critical": 0.01,
    },
    "medium": {
        "minor": 0.19,
        "moderate": 0.46,
        "major": 0.28,
        "business_critical": 0.07,
    },
    "high": {
        "minor": 0.04,
        "moderate": 0.28,
        "major": 0.46,
        "business_critical": 0.22,
    },
    "critical": {
        "minor": 0.01,
        "moderate": 0.08,
        "major": 0.35,
        "business_critical": 0.56,
    },
}

SEVERITY_SECURITY_OVERRIDES = {"minor": 0.0}

SEVERITY_SECURITY_ADJUSTMENTS = {
    "major": 0.16,
    "business_critical": 0.10,
}

SEVERITY_TICKET_CATEGORY_ADJUSTMENTS = {
    **{
        ticket_category: {"moderate": 0.08}
        for ticket_category in ("invoice_issue", "purchase_request", "privacy_request")
    },
    **{
        ticket_category: {
            "moderate": 0.06,
            "major": 0.10,
        }
        for ticket_category in (
            "developer_tooling",
            "support_platform",
            "data_platform",
            "pay_cycle_issue",
        )
    },
}

IMPACT_SCOPE_BASE_WEIGHTS = {
    "single_user": 0.50,
    "single_team": 0.22,
    "single_department": 0.15,
    "multi_department": 0.10,
    "company_wide": 0.03,
}

IMPACT_SCOPE_TICKET_CATEGORY_ADJUSTMENTS = {
    **{
        ticket_category: {"single_user": 0.18}
        for ticket_category in (
            "access_request",
            "account_issue",
            "software_issue",
            "reimbursement",
        )
    },
    **{
        ticket_category: {
            "single_team": 0.10,
            "single_department": 0.08,
        }
        for ticket_category in ("purchase_request", "workspace_request", "invoice_issue")
    },
    **{
        ticket_category: {
            "single_department": 0.12,
            "multi_department": 0.10,
        }
        for ticket_category in ("support_platform", "data_platform", "regional_compliance")
    },
    **{
        ticket_category: {"single_team": 0.10}
        for ticket_category in ("crm_support", "marketing_tools", "analytics_access")
    },
}

IMPACT_SCOPE_SECURITY_ADJUSTMENTS = {
    "single_user": -0.18,
    "single_department": 0.10,
    "multi_department": 0.12,
    "company_wide": 0.08,
}

IMPACT_SCOPE_URGENCY_ADJUSTMENTS = {
    "critical": {
        "single_user": -0.16,
        "multi_department": 0.10,
        "company_wide": 0.10,
    },
    "high": {
        "single_team": 0.08,
        "single_department": 0.06,
    },
}

WRITING_STYLE_BASE_WEIGHTS = {
    "concise": 0.20,
    "formal": 0.20,
    "informal": 0.18,
    "frustrated": 0.12,
    "technical": 0.18,
    "non_native_english": 0.12,
}

WRITING_STYLE_DEPARTMENT_ADJUSTMENTS = {
    **{
        department: {
            "formal": 0.02,
            "technical": 0.14,
        }
        for department in ("IT", "Security", "Engineering", "Data")
    },
    **{department: {"formal": 0.06} for department in ("Finance", "Legal", "Travel", "Payroll")},
}

WRITING_STYLE_HIGH_PRESSURE_ADJUSTMENTS = {
    "frustrated": 0.12,
}

WRITING_STYLE_REQUESTER_ROLE_ADJUSTMENTS = {
    **{requester_role: {"formal": 0.12} for requester_role in ("director", "executive_assistant")},
    **{requester_role: {"informal": 0.08} for requester_role in ("new_hire", "intern")},
}

CLARITY_LEVEL_BASE_WEIGHTS = {
    "clear": 0.34,
    "moderately_clear": 0.36,
    "vague": 0.10,
    "missing_some_details": 0.20,
}

CLARITY_LEVEL_WRITING_STYLE_OVERRIDES = {
    "technical": {"vague": 0.04},
}

CLARITY_LEVEL_WRITING_STYLE_ADJUSTMENTS = {
    "technical": {"clear": 0.18},
    "formal": {"clear": 0.12},
    "frustrated": {
        "missing_some_details": 0.14,
        "vague": 0.06,
    },
    "concise": {"moderately_clear": 0.10},
    "non_native_english": {
        "clear": -0.06,
        "missing_some_details": 0.12,
    },
}

CLARITY_LEVEL_REQUESTER_ROLE_ADJUSTMENTS = {
    **{requester_role: {"missing_some_details": 0.08} for requester_role in ("intern", "new_hire")},
}

DIFFICULTY_PROFILES: dict[str, DifficultyProfileConfig] = {
    "easy": {
        "name": "easy",
        "writing_style_adjustments": {
            "concise": 0.06,
            "formal": 0.08,
            "frustrated": -0.05,
            "non_native_english": -0.04,
        },
        "clarity_adjustments": {
            "clear": 0.18,
            "moderately_clear": 0.08,
            "vague": -0.08,
            "missing_some_details": -0.12,
        },
        "opener_probability": 0.62,
        "explicit_system_probability": 0.95,
        "explicit_resolution_probability": 1.0,
        "context_probability": 0.78,
        "progress_probability": 0.68,
        "pressure_probability": 0.55,
        "indirect_issue_probability": 0.05,
        "symptom_first_probability": 0.15,
        "sentence_pattern_weights": {
            "standard": 0.65,
            "context_first": 0.18,
            "status_first": 0.1,
            "pressure_first": 0.02,
            "compressed": 0.05,
        },
    },
    "medium": {
        "name": "medium",
        "writing_style_adjustments": {},
        "clarity_adjustments": {},
        "opener_probability": 0.45,
        "explicit_system_probability": 0.78,
        "explicit_resolution_probability": 0.92,
        "context_probability": 0.62,
        "progress_probability": 0.5,
        "pressure_probability": 0.45,
        "indirect_issue_probability": 0.15,
        "symptom_first_probability": 0.28,
        "sentence_pattern_weights": {
            "standard": 0.42,
            "context_first": 0.18,
            "status_first": 0.16,
            "pressure_first": 0.08,
            "compressed": 0.16,
        },
    },
    "hard": {
        "name": "hard",
        "writing_style_adjustments": {
            "informal": 0.04,
            "frustrated": 0.08,
            "non_native_english": 0.08,
            "technical": -0.03,
        },
        "clarity_adjustments": {
            "clear": -0.1,
            "moderately_clear": -0.05,
            "vague": 0.08,
            "missing_some_details": 0.12,
        },
        "opener_probability": 0.24,
        "explicit_system_probability": 0.45,
        "explicit_resolution_probability": 0.65,
        "context_probability": 0.42,
        "progress_probability": 0.36,
        "pressure_probability": 0.32,
        "indirect_issue_probability": 0.36,
        "symptom_first_probability": 0.45,
        "sentence_pattern_weights": {
            "standard": 0.25,
            "context_first": 0.18,
            "status_first": 0.15,
            "pressure_first": 0.15,
            "compressed": 0.27,
        },
    },
}

BANNED_INTERNAL_PROCESS_PHRASES = (
    "pending approval",
    "manager approval",
    "finance approval",
    "security review",
    "security approval",
    "waiting for another team",
    "system sync",
    "multi-step approval",
    "company-wide impact",
    "business critical",
    "workflow stage",
    "dependency type",
    "impact scope",
)
