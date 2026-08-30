export type CrimeFamilyCode = `F${string}`;
export type CrimeSubtypeCode = `F${string}.${string}`;

type FamilyDefinition = {
  code: CrimeFamilyCode;
  key: string;
  label: string;
  subtypes: readonly [code: CrimeSubtypeCode, key: string][];
};

const family = (
  code: CrimeFamilyCode,
  key: string,
  label: string,
  subtypes: FamilyDefinition["subtypes"],
): FamilyDefinition => ({ code, key, label, subtypes });

export const CRIME_FAMILIES = [
  family("F01", "homicide", "Homicide", [
    ["F01.02", "manslaughter"],
    ["F01.03", "murder_homicide"],
  ]),
  family("F02", "sexual_offense", "Sexual Offense", [
    ["F02.01", "fondling"],
    ["F02.02", "incest"],
    ["F02.03", "other_sex_offense"],
    ["F02.04", "rape"],
    ["F02.05", "sexual_assault"],
    ["F02.06", "statutory_rape"],
    ["F02.07", "child_sexual_exploitation"],
  ]),
  family("F03", "robbery", "Robbery", [["F03.01", "robbery"]]),
  family("F04", "assault", "Assault", [
    ["F04.01", "aggravated_assault"],
    ["F04.02", "simple_assault"],
    ["F04.03", "reckless_endangerment"],
  ]),
  family("F05", "kidnapping_trafficking", "Kidnapping / Trafficking", [
    ["F05.01", "human_trafficking"],
    ["F05.02", "human_trafficking_commercial_sex"],
    ["F05.03", "human_trafficking_involuntary_servitude"],
    ["F05.04", "kidnapping_abduction"],
    ["F05.05", "unlawful_restraint"],
  ]),
  family("F06", "intimidation_stalking", "Intimidation / Stalking", [
    ["F06.01", "intimidation_harassment"],
    ["F06.02", "stalking"],
    ["F06.03", "privacy_surveillance_offense"],
    ["F06.04", "terroristic_threat_hoax"],
  ]),
  family("F07", "burglary", "Burglary", [["F07.01", "burglary"]]),
  family("F08", "larceny_theft", "Larceny / Theft", [
    ["F08.01", "other_larceny_theft"],
    ["F08.02", "pocket_picking"],
    ["F08.03", "purse_snatching"],
    ["F08.04", "shoplifting"],
    ["F08.05", "theft_from_building"],
    ["F08.06", "theft_from_vehicle"],
    ["F08.07", "theft_vehicle_parts"],
  ]),
  family("F09", "motor_vehicle_theft", "Motor Vehicle Theft", [["F09.01", "motor_vehicle_theft"]]),
  family("F10", "arson", "Arson", [["F10.01", "arson"]]),
  family("F11", "vandalism_property_damage", "Vandalism / Property Damage", [
    ["F11.01", "vandalism_property_damage"],
  ]),
  family("F12", "fraud_financial", "Fraud / Financial", [
    ["F12.01", "bad_checks"],
    ["F12.02", "bribery"],
    ["F12.03", "embezzlement"],
    ["F12.04", "extortion_blackmail"],
    ["F12.05", "forgery_counterfeiting"],
    ["F12.06", "fraud"],
    ["F12.07", "identity_cyber_fraud"],
  ]),
  family("F13", "stolen_property", "Stolen Property", [["F13.01", "stolen_property"]]),
  family("F14", "drug_offense", "Drug Offense", [
    ["F14.01", "drug_equipment"],
    ["F14.02", "drug_narcotic"],
  ]),
  family("F15", "weapons_offense", "Weapons Offense", [["F15.01", "weapons_violation"]]),
  family("F16", "prostitution_commercial_sex", "Prostitution / Commercial Sex", [
    ["F16.01", "promoting_prostitution"],
    ["F16.02", "prostitution"],
    ["F16.03", "purchasing_prostitution"],
  ]),
  family("F17", "public_order", "Public Order", [
    ["F17.01", "curfew_loitering_vagrancy"],
    ["F17.02", "disorderly_conduct"],
    ["F17.03", "gambling"],
    ["F17.04", "liquor_law"],
    ["F17.05", "obscenity_indecency"],
    ["F17.06", "obstruction_resisting"],
    ["F17.07", "other_public_order"],
    ["F17.08", "public_intoxication"],
    ["F17.09", "trespass"],
    ["F17.10", "court_order_violation"],
    ["F17.11", "emergency_service_interference"],
    ["F17.12", "escape_bail_jumping"],
    ["F17.13", "evading_resisting"],
    ["F17.14", "false_reporting"],
    ["F17.15", "municipal_code_violation"],
    ["F17.16", "obstruction_of_justice"],
    ["F17.17", "public_administration_offense"],
    ["F17.18", "public_camping"],
    ["F17.19", "public_health_environmental_violation"],
    ["F17.20", "solicitation_panhandling"],
    ["F17.21", "tax_business_regulation_violation"],
    ["F17.22", "transit_rule_violation"],
  ]),
  family("F18", "family_child_offense", "Family / Child Offense", [
    ["F18.01", "family_child_offense"],
    ["F18.02", "family_offense_nonviolent"],
    ["F18.03", "family_order_violation"],
    ["F18.04", "child_custody_interference"],
  ]),
  family("F19", "traffic_offense", "Traffic Offense", [
    ["F19.01", "dui"],
    ["F19.02", "traffic_violation"],
    ["F19.03", "hit_and_run"],
    ["F19.04", "street_racing"],
    ["F19.05", "vehicle_document_violation"],
  ]),
  family("F20", "other_criminal", "Other Criminal", [
    ["F20.01", "animal_cruelty"],
    ["F20.02", "not_reportable_nibrs"],
    ["F20.03", "other_criminal"],
    ["F20.04", "registration_violation"],
    ["F20.05", "ritualism"],
    ["F20.06", "school_offense"],
    ["F20.07", "conspiracy_organized_crime"],
    ["F20.08", "criminal_instruments"],
    ["F20.09", "terrorism_support"],
  ]),
] as const satisfies readonly FamilyDefinition[];

const labelForKey = (key: string) =>
  key
    .split("_")
    .map((word) =>
      word === "dui" || word === "nibrs"
        ? word.toUpperCase()
        : word[0].toUpperCase() + word.slice(1),
    )
    .join(" ");

export const CRIME_SUBTYPES = CRIME_FAMILIES.flatMap((item) =>
  item.subtypes.map(([subtypeCode, subtypeKey]) => ({
    familyCode: item.code,
    familyKey: item.key,
    familyLabel: item.label,
    subtypeCode,
    subtypeKey,
    subtypeLabel: labelForKey(subtypeKey),
  })),
);

export const CRIME_FAMILY_BY_CODE = new Map(CRIME_FAMILIES.map((item) => [item.code, item]));
export const CRIME_SUBTYPE_BY_CODE = new Map(
  CRIME_SUBTYPES.map((item) => [item.subtypeCode, item]),
);

export type MarkClassMetadata = (typeof CRIME_SUBTYPES)[number] & { classId: number };

// The subtype order was reconciled entry-for-entry with the authoritative
// crimenet_mark_class_labels_v1 artifact at C:\crimenet\class_labels.json.
// XGBoost output identity is always resolved through class_id, never through
// response-array position or alphabetic label order.
export const MARK_CLASS_BY_ID = new Map<number, MarkClassMetadata>(
  CRIME_SUBTYPES.map((item, classId) => [classId, { classId, ...item }]),
);
