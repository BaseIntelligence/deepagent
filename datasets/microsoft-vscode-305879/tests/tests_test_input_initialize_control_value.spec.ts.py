import { provideZoneChangeDetection } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { By } from '@angular/platform-browser';
import { Component } from '@angular/core';

import { provideNzIconsTesting } from 'ng-zorro-antd/icon/testing';
import { NzInputDirective } from '../components/input/input.directive';
import { NzInputModule } from '../components/input/input.module';

describe('NzInputDirective control initialization', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideNzIconsTesting(), provideZoneChangeDetection()]
    });
  });

  it('initializes directive value from an existing form control value', () => {
    const fixture: ComponentFixture<TestInputControlInitComponent> = TestBed.createComponent(TestInputControlInitComponent);
    fixture.detectChanges();

    const inputDebugElement = fixture.debugElement.query(By.directive(NzInputDirective));
    const inputDirective = inputDebugElement.injector.get(NzInputDirective);

    expect(fixture.componentInstance.formControl.value).toBe('abc');
    expect(inputDirective.value()).toBe('abc');
  });
});

@Component({
  imports: [ReactiveFormsModule, NzInputModule],
  template: `<input nz-input [formControl]="formControl" />`
})
class TestInputControlInitComponent {
  formControl = new FormControl('abc');
}
