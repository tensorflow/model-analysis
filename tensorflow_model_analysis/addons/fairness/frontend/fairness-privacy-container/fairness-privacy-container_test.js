/**
 * Copyright 2019 Google LLC
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

suite('fairness-privacy-container tests', () => {
  let privacyContainer;

  const OMITTED_SLICES = [
    'Slice:1', 'Slice:2', 'Slice:3'
  ];

  test('CheckOmittedSlicesList', done => {
    privacyContainer = fixture('test-fixture');

    const FillData = () => {
      privacyContainer.omittedSlices = OMITTED_SLICES;
      setTimeout(CheckOmittedSlicesListValue, 0);
    };

    const CheckOmittedSlicesListValue = () => {
      let listbox = privacyContainer.$['omitted-slices-list'];
      assert.equal(listbox.items.length, 3);
      assert.isTrue(listbox.items[0].textContent.includes('Slice:1'));
      assert.isTrue(listbox.items[1].textContent.includes('Slice:2'));
      assert.isTrue(listbox.items[2].textContent.includes('Slice:3'));
      done();
    };

    setTimeout(FillData, 0);
  });

  test('CheckIfPrivacyDialogBoxOpensUp', done => {
    privacyContainer = fixture('test-fixture');

    const CheckIfDialogBloxIsClosed = () => {
      let dialogBox = privacyContainer.$['privacy-dialog'];
      assert.isFalse(dialogBox.opened);
      setTimeout(TapClickHereButton, 0);
    };

    const TapClickHereButton = () => {
      privacyContainer.$['paper-button'].fire('tap');
      setTimeout(CheckIfDialogBloxIsOpened, 0);
    };

    const CheckIfDialogBloxIsOpened = () => {
      let dialogBox = privacyContainer.$['privacy-dialog'];
      assert.isTrue(dialogBox.opened);
      done();
    };

    setTimeout(CheckIfDialogBloxIsClosed, 0);
  });
});
