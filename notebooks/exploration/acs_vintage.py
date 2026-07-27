from datetime import date

# Format:

calendar_data = [
    # acs_vintage, acs_release_date, tiger_line_year, tract_definition_vintage
    (2012, date(2013, 12, 17), 2012, 2010),
    (2013, date(2014, 12, 4),  2013, 2010),
    (2014, date(2015, 12, 3),  2014, 2010),
    (2015, date(2016, 12, 8),  2015, 2010),
    (2016, date(2017, 12, 7),  2016, 2010),
    (2017, date(2018, 12, 6),  2017, 2010),
    (2018, date(2019, 12, 19), 2018, 2010),
    (2019, date(2020, 12, 10), 2019, 2010),
    (2020, date(2022, 3, 17),  2020, 2020),
    (2021, date(2022, 12, 8),  2021, 2020),
    (2022, date(2023, 12, 7),  2022, 2020),
    (2023, date(2024, 12, 12), 2023, 2020),
    (2024, date(2026, 1, 29),  2024, 2020),
]

schema = """
    acs_vintage INT,
    acs_release_date DATE,
    tiger_line_year INT,
    tract_definition_vintage INT
"""

acs_calendar_df = spark.createDataFrame(
    calendar_data,
    schema=schema,
)

(
    acs_calendar_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "crimenet_dev.silver.acs_vintage_calendar"
    )
)

print(
    "Successfully created "
    "crimenet_dev.silver.acs_vintage_calendar"
)

display(
    acs_calendar_df.orderBy("acs_vintage")
)