# TBI-TRACT Shiny entrypoint
#
# The full application implementation is preserved in app_core.R. This wrapper
# provides a compatibility alias for the numeric input constructor and then
# returns the Shiny app object defined by app_core.R.

suppressPackageStartupMessages(library(shiny))

# Compatibility alias: the implementation historically called numberInput(),
# while Shiny's actual constructor is numericInput(). Keeping the alias here
# avoids altering the locked application logic and prevents renderUI failures.
numberInput <- shiny::numericInput

source("app_core.R", local = TRUE)$value
