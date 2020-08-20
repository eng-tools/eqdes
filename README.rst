README
======

This package contains solvers for direct displacement-based design

Major revision at version 0.3.0 - this package is still in alpha.

Function and method naming conventions
--------------------------------------


* `calc_<property>`: A function that calculates a property,
  if the calculation has multiple different implementations, then include the citation as
  author and year as well `calc_<property>_<author>_<year>`
* `design_<system-to-be-designed>` A function that implements a design procedure and returns a designed object
* `assess_<system-to-be-designed>` A function that implements an assessment procedure and returns a assessed object

Setting up tests
----------------

If using PyCharm:
 1. make sure the navigation bar is showing (View>Navigation bar)
 2. Edit Configuration (top right)
 3. Click the plus symbol in the pop up window
 4. Select pytest from the drop down menu
 5. point it to the tests folder
 6. Call the configuration "All tests"


What is this repository for?
----------------------------

* Quick summary
* Version

How do I get set up?
--------------------

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

Contribution guidelines
-----------------------

.. Standard

* Writing tests
* Code review
* Other guidelines
