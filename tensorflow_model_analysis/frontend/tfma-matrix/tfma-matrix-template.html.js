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
    --cell-width: 50px;
    --cell-height: 30px;
  }
  .matrix-row {
    display: flex;
  }
  :host([expanded]) .cell, :host([expanded]) .header, :host([expanded]) .widget {
    width: var(--cell-width);
    height: var(--cell-height);
    line-height:var(--cell-height);
    font-size: 12px;
    text-overflow: ellipsis;
    text-align: center;
  }

  .sizer {
    float: left;
    position: relative;
  }

  .missing {
    background-color: #ccc;
  }

  .cell, .header {
    width: 3px;
    height: 3px;
    overflow: hidden;
    cursor: pointer;
  }

  .label {
    position: absolute;
    width: 100%;
    height: var(--cell-height);

    /**
     * Make the nodes helping to size and position invisible so that they do not block the click
     * events on the headers / cells.
     */
    visibility: hidden;
  }

  .label .rowLabel, .label .columnLabel {
    text-align: center;
    font-size: 12px;
  }

  .column-label-sizer {
    margin-left: var(--cell-width);
  }

  .label .columnLabel {
    position: relative;
    text-align: center;
    top: -20px;
    visibility: visible;
    height: 16px;
    width:100%;
  }

  .row-label-anchor {
    width: 60%;
    position:relative;
    /** The height of the older sibling div, column label, is 16px. */
    top:-16px;
  }

  .row-label-sizer {
    float: left;
    width: 100%;
    /**
     * When expanded, the cell height is 60% of cell width. After rotation,
     * we need to shift the row label by the full height of the matrix so
     * we set a margin top of 60%.
     */
    margin-top: 60%;
    position: absolute;
    top: 0;
  }

  .label .rowLabel {
    transform: rotate(-90deg);
    transform-origin: bottom left;
    position: relative;
    left: -10px;

    /**
     * When expanded, the cell height is 60% of cell width. To center the
     * label text, set the width of the container div to 60% so that it
     * will match the height of the matrix after rotation.
     */
    width: 60%;

    /**
     * An extra offset to make the label appear more centered.
     */
    top: 15px;

    visibility: visible;
  }

  :host([expanded]) #root, #details {
    margin: 24px;
    clear: both;
  }

  .c0 {
    /** Red */
    background-color: #F44336;
  }
  .c1 {
    /** Purple */
    background-color: #9C27B0;
  }
  .c2 {
    /** Deep purple */
    background-color: #673AB7;
  }
  .c3 {
    /** Indigo */
    background-color: #3F51B5;
  }
  .c4 {
    /** Blue */
    background-color: #2196F3;
  }
  .c5 {
    /** Light blue*/
    background-color: #03A9F4;
  }
  .c6 {
    /** Cyan */
    background-color: #00BCD4;
  }
  .c7 {
    /** Teal */
    background-color: #009688;
  }
  .c8 {
    /** Green */
    background-color: #4CAF50;
  }
  .c9 {
    /** Light Green */
    background-color: #8BC34A;
  }
  .c10 {
    /** Lime */
    background-color: #CDDC39;
  }
  .c11 {
    /** Yellow */
    background-color: #FFEB3B;
  }
  .c12 {
    /** Amber */
    background-color: #FFC107;
  }
  .c13 {
    /** Orange */
    background-color: #FF9800;
  }
  .c14 {
    /** Deep Orange */
    background-color: #FF5722;
  }
  .c15 {
    /** Pink */
    background-color: #E91E63;
  }

  :host(:not([expanded])) #root {
    min-height: 30px;
    transform-origin: top left;
  }

  :host(:not([expanded])) .s1 {
    transform: scale3d(10, 10, 1);
  }

  :host(:not([expanded])) .s2 {
    transform: scale3d(5, 5, 1);
  }

  :host(:not([expanded])) .s3 {
    transform: scale3d(3.33, 3.33, 1);
  }

  :host(:not([expanded])) .s4 {
    transform: scale3d(2.5, 2.5, 1);
  }

  :host(:not([expanded])) .s5 {
    transform: scale3d(2, 2, 1);
  }

  :host(:not([expanded])) .s6 {
    transform: scale3d(1.66, 1.66, 1);
  }

  :host(:not([expanded])) .s7 {
    transform: scale3d(1.42, 1.42, 1);
  }

  :host(:not([expanded])) .s8 {
    transform: scale3d(1.25, 1.25, 1);
  }

  :host(:not([expanded])) .s9 {
    transform: scale3d(1.11, 1.11, 1);
  }

  .cell.b0 {
    background-color: rgb(240, 240, 240);
  }
   .cell.b1 {
    background-color: rgb(226, 229, 235);
  }
  .cell.b2 {
    background-color: rgb(211, 219, 231);
  }
  .cell.b3 {
    background-color: rgb(197, 208, 226);
  }
  .cell.b4 {
    background-color: rgb(183, 198, 221);
  }
  .cell.b5 {
    background-color: rgb(168, 187, 216);
  }
  .cell.b6 {
    background-color: rgb(154, 177, 212);
  }
  .cell.b7 {
    background-color: rgb(139, 166, 207);
  }
  .cell.b8 {
    background-color: rgb(125, 156, 202);
  }
  .cell.b9 {
    background-color: rgb(111, 145, 197);
  }
  .cell.b10 {
    background-color: rgb(96, 134, 193);
  }
  .cell.b11 {
    background-color: rgb(82, 124, 188);
  }
  .cell.b12 {
    background-color: rgb(68, 113, 183);
  }
  .cell.b13 {
    background-color: rgb(53, 103, 178);
  }
  .cell.b14 {
    background-color: rgb(39, 92, 174);
  }
  .cell.b15 {
    background-color: rgb(24, 82, 169);
  }
  .cell.b16 {
    background-color: rgb(10, 71, 164);
  }

  .white-text {
    color: white;
  }
</style>
<div id="root" class$="s[[rowNames_.length]]">
  <div class="sizer">
    <template is="dom-repeat" items="[[matrix_]]" as="row" initial-count="10">
      <div class="matrix-row">
        <template is="dom-repeat" items="[[row]]" as="cell" initial-count="10">
          <template is="dom-if" if="[[cell.widget]]">
            <div class="widget"></div>
          </template>
          <template is="dom-if" if="[[cell.label]]">
            <div class="label">
              <div class="column-label-sizer">
                <div class="columnLabel">[[columnLabel]]</div>
              </div>
              <div class="row-label-anchor">
                <div class="row-label-sizer">
                  <div class="rowLabel">[[rowLabel]]</div>
                </div>
              </div>
            </div>
          </template>
          <template is="dom-if" if="[[cell.header]]">
            <a title="[[cell.name]]">
              <div class$="[[cell.cssClass]]" on-tap="headerTapped_">
                [[cell.name]]
              </div>
            </a>
          </template>
          <template is="dom-if" if="[[cell.cell]]">
            <a title$="[[getCellTitle_(expanded, cell)]]">
              <div class$="cell [[cell.cssClass]]" style$="[[cell.style]]" on-tap="cellTapped_">
                [[cell.value]]
              </div>
            </a>
          </template>
          <template is="dom-if" if="[[cell.missing]]">
              <a title="No value">
                <div on-tap="cellTapped_" class="cell missing">
                </div>
              </a>
          </template>
        </template>
      </div>
    </template>
    <template is="dom-if" if="[[tooMany_(rowNames_, columnNames_)]]">
      <a title="Too many classes for visualization.">Omitted</a>
    </template>
  </div>
</div>
<div id="details">
  <template is="dom-if" if="[[selectedCell_]]">
    <div id="cell-details">
      [[selectedCell_.details]]
    </div>
  </template>
</div>
`;
export {template};
