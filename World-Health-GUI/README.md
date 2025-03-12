Overview

World Health GUI is a Python-based graphical user interface (GUI) application that allows users to explore global health data interactively. The application provides various exploration modes, enabling users to query health-related information by health area, country, year, or region.

Features

1. Main GUI Window
Displays a window titled "World Health Database".
Provides a drop-down menu (combobox) for selecting the exploration mode:
By Health Area
By Country
By Year
By Region
Clicking the "Explore" button triggers the selected exploration mode.

2. Exploration Modes
   
--> By Health Area

Opens a new window where users can select specific health topics, including:
Child Health
Maternal Health
Death Factors
Life Expectancy
Vaccination
After selecting a health area and clicking "Query Data", the relevant data is displayed.
Users can filter results by country using a combobox and filter button.

--> By Country

Opens a window with a listbox to select a country.
Clicking "Explore" displays the country-specific health data in a new window.

--> By Year

Opens a new window where users can select a specific year.
Clicking "Explore" retrieves and displays health data for that year.
Users can filter the data by country or region using comboboxes and filter buttons.

--> By Region

Opens a window where users can select a region from a list of subregions.
Clicking "Explore" displays aggregated health data for the selected region.
