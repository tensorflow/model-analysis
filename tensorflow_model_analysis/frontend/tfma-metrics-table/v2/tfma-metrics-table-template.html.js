/**
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
const template = /** @type {!HTMLTemplateElement} */(document.createElement('template'));
template.innerHTML = `
<style>
  :host {
    display: block;
    box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14);
  }

  google-chart {
    width: auto;
    height: auto;
  }
</style>
<style>
  /**
   * HACK: Since there is no way for us to use CSS to style the table under
   * shadow dom, we will use JS to copy the CSS styles directly instead.
   * All rules within this style tag are for the hack and should have the
   * prefix "#hack ". Using the prefix allows us to easily identify them (since
   * the vulcanization process merges style tags) and ensures they do not get
   * applied to elements inside this component.
   */
  #hack .google-visualization-table * {
    font-family: 'Roboto','Noto','Helvetica Neue',Helvetica,Arial,sans-serif;
    font-weight: 400;
    font-size: 12px;
    text-align: left;
  }

  #hack .google-visualization-table-table th.google-visualization-table-th {
    font-weight: 500;
    text-overflow: ellipsis;
    color: rgba(0,0,0,0.54);
    border-right: none;
    padding: 0 24px 0 24px;
    text-align:center;
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.09),
                0 2px 0 rgba(0, 0, 0, 0.03),
                0 3px 0 rgba(0, 0, 0, 0.01),
                0 4px 0 rgba(0, 0, 0, 0.005);
  }

  #hack .google-visualization-table-table td.google-visualization-table-td {
    font-size: 13px;
    color: rgba(0,0,0,0.87);
    border-bottom: 1px solid #e3e3e3;
    border-right: none;
    padding: 0 24px 0 24px;
  }

  #hack .google-visualization-table-tr-head {
    height: 56px;
    background: none;
  }

  #hack tbody > tr {
    line-height: 2.2;
  }

  #hack tbody > tr > td:nth-child(odd),
  #hack thead > tr > th:nth-child(odd)  {
    background-color: rgba(0, 0, 0, 0.05);
  }


  #hack .sort-descending > span.google-visualization-table-sortind::after {
    content: "➔";
    font-size: 13px;
    display: inline-block;
    transform: rotate(90deg);
    color: rgba(0, 0, 0, 0.87);
  }

  #hack .sort-ascending > span.google-visualization-table-sortind::after {
    content: "➔";
    font-size: 13px;
    display: inline-block;
    transform: rotate(-90deg);
    color: rgba(0, 0, 0, 0.87);
  }

  #hack .google-visualization-table-div-page {
    background-color: white;
    position: relative;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.14);
    border: none;
  }

  #hack tr.google-visualization-table-tr-sel,
  #hack tr.google-visualization-table-tr-sel.google-visualization-table-tr-over {
    background-color: #eee;
  }

  #hack tr.google-visualization-table-tr-over {
    background-color: #e9e9e9;
  }

  #hack .google-visualization-table-tr-odd {
    background-color: white;
  }

  #hack a.google-visualization-table-page-number:hover {
    font-weight: bold;
  }

  #hack a.google-visualization-table-page-number {
    padding: 4px;
    background-color: white;
    border: 1px solid rgba(0,0,0,.2);
    text-align: center;
    font-size: 11px;
    color: rgba(0,0,0,.87);
  }

  #hack a.google-visualization-table-page-number.current {
    font-size: 12px;
  }

  #hack .goog-custom-button-hover .goog-custom-button-inner-box,
  #hack .goog-custom-button-hover .goog-custom-button-outer-box {
    border-color: rgba(0,0,0,.3) !important; /* Overwrite google-chart important */
  }

  #hack .goog-inline-block.goog-custom-button-outer-box {
    background-color: white;
    border-color: rgba(0,0,0,.2);
  }

  #hack .google-visualization-table-div-page .goog-inline-block.goog-custom-button-inner-box {
    padding: 4px;
    color: rgba(0,0,0,.4);
  }

  /**
   * Hides .subheader by making its height 0. Note that we did not use
   * display:none to ensure the sub-columns under each subheader stay aligned.
   */
  #hack table tr:nth-child(n+2) .subheader,
  #hack table tr:first-child metric-diff #second-container .subheader {
    height: 0;
    overflow: hidden;
  }

  /**
   * Make sure titles in subheader have consistent styles.
   */
  #hack table tr td .subheader.title {
    font-weight: 500;
    color: rgba(0,0,0,0.54);
  }
</style>

<google-chart type="table" options="[[options_]]" data="[[plotData_]]" selection="{{selection}}"
              id="table" events="[[chartEvents_]]" on-google-chart-page="onPage_"
              on-google-chart-sort="onSort_">
</google-chart>
`;
export {template};
