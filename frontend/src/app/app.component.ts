import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { BaseChartDirective } from 'ng2-charts';
import { HttpClient } from '@angular/common/http';
import { ChartData, ChartType, ChartOptions } from 'chart.js';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, BaseChartDirective, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  public chartType: ChartType = 'line';
  public chartData: ChartData<'line'> = { datasets: [] };

  public activeTab: string = 'resumen';
  public activeRange: string = '';
  public rangeButtons: string[] = [
    '1d',
    '5d',
    '1mo',
    '3mo',
    '6mo',
    '1y',
    '2y',
    '5y',
    '10y',
    'max',
  ];

  public startDate: string = '';
  public endDate: string = '';
  public futureDays: number = 1;
  public yesterday: Date = new Date();
  public earliestDay: Date = new Date('1980-01-01');

  // Datos de cada sección
  public chartDataResumen: ChartData<'line'> = { datasets: [] };
  public chartDataNormal: ChartData<'line'> = { datasets: [] };
  public chartDataRecursiva: ChartData<'line'> = { datasets: [] };

  // Fechas de cada sección
  public datesResumen: Date[] = [];
  public datesNormal: Date[] = [];
  public datesRecursiva: Date[] = [];

  // Límites de cada sección
  public xAxisMinResumen: Date | null = null;
  public xAxisMaxResumen: Date | null = null;
  public xAxisMinNormal: Date | null = null;
  public xAxisMaxNormal: Date | null = null;
  public xAxisMinRecursiva: Date | null = null;
  public xAxisMaxRecursiva: Date | null = null;

  public errorMessage: string = '';
  public isLoading: boolean = false;
  private destroy$ = new Subject<void>();

  selectRange(label: string): void {
    this.activeRange = label;
    this.getHistory(label);
  }

  public chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { top: 0, right: 40, bottom: 20, left: 20 },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          displayFormats: { day: 'dd/MM/yyyy' },
        },
        ticks: {
          source: 'auto',
          autoSkip: true,
          color: 'white',
          minRotation: 45,
          padding: 10,
        },
        grid: { display: false },
        bounds: 'ticks',
        offset: true,
      },
      y: {
        type: 'linear',
        title: {
          display: true,
          text: 'Precio (USD)',
          color: 'white',
        },
        ticks: {
          color: 'white',
          padding: 20,
        },
        grid: {
          color: '#4a4a4a',
          display: true,
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: { color: 'white' },
      },
    },
  };

  private updateXAxisLabels(dates: Date[]): void {
    const totalDays =
      (dates[dates.length - 1].getTime() - dates[0].getTime()) /
      (1000 * 3600 * 24);

    let timeUnit: 'day' | 'week' | 'month' | 'year' = 'day';
    let displayFormats = {};
    let maxTicksLimit: number | undefined = undefined;

    if (totalDays < 7) {
      timeUnit = 'day';
      displayFormats = { day: 'dd/MM/yyyy' };
      maxTicksLimit = dates.length;
    } else if (totalDays >= 7 && totalDays <= 30) {
      timeUnit = 'day';
      displayFormats = { day: 'dd/MM/yyyy' };
      maxTicksLimit = Math.ceil(totalDays / 3);
    } else if (totalDays > 30 && totalDays <= 90) {
      timeUnit = 'week';
      displayFormats = { week: 'dd/MM/yyyy' };
      maxTicksLimit = Math.ceil(totalDays / 7);
    } else if (totalDays > 90 && totalDays <= 720) {
      timeUnit = 'month';
      displayFormats = { month: 'MM/yyyy' };
      maxTicksLimit = Math.ceil(totalDays / 60);
    } else {
      timeUnit = 'year';
      displayFormats = { year: 'yyyy' };
      maxTicksLimit = undefined;
    }

    const xScale = this.chartOptions.scales!['x'] as any;
    xScale.time.unit = timeUnit;
    xScale.time.displayFormats = displayFormats;
    xScale.ticks.autoSkip = true;
    xScale.ticks.source = 'data';
    xScale.ticks.maxTicksLimit = maxTicksLimit;

    if (dates.length === 1) {
      const onlyDate = dates[0].getTime();
      xScale.min = onlyDate - 1000 * 3600 * 24;
      xScale.max = onlyDate + 1000 * 3600 * 24;
    } else {
      xScale.min = dates[0].getTime();
      xScale.max = dates[dates.length - 1].getTime();
    }

    xScale.bounds = 'ticks';
    xScale.offset = true;

    this.chartOptions = { ...this.chartOptions };
  }

  private getPointRadius(dates: Date[]): number {
    const totalDays =
      (dates[dates.length - 1].getTime() - dates[0].getTime()) /
      (1000 * 3600 * 24);
    return totalDays < 365 ? 2 : 0;
  }

  /**
   * Validar fechas (para predicción pasada)
   */
  validateDates(): boolean {
    const today = new Date().toISOString().split('T')[0];

    if (!this.startDate || !this.endDate) {
      alert('Por favor, seleccione ambas fechas.');
      return false;
    }

    if (new Date(this.startDate) < this.earliestDay) {
      alert('La fecha de inicio debe ser posterior al 01/01/1980.');
      this.startDate = this.earliestDay.toISOString().split('T')[0];
      return false;
    }

    if (this.startDate >= this.endDate) {
      alert('La fecha de inicio debe ser anterior a la fecha de fin.');
      const endDateObj = new Date(this.endDate);
      endDateObj.setDate(endDateObj.getDate() - 1);
      this.startDate = endDateObj.toISOString().split('T')[0];
      return false;
    }

    if (this.endDate >= today) {
      alert('La fecha de fin debe ser anterior a la fecha actual.');
      this.endDate = this.yesterday.toISOString().split('T')[0];
      return false;
    }

    return true;
  }

  /**
   * Validar número de días futuros (para predicción recursiva)
   */
  validateFutureDays(): boolean {
    if (!this.futureDays || this.futureDays <= 0 || this.futureDays > 180) {
      alert(
        'La cantidad de días futuros debe estar comprendida entre 1 y 180.'
      );
      return false;
    }
    return true;
  }

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.getHistory('6mo');
    this.setActiveTab('resumen');
    this.yesterday.setDate(this.yesterday.getDate() - 1);
    this.endDate = this.yesterday.toISOString().split('T')[0];
  }

  /**
   * Obtener y cargar histórico completo
   */
  getHistory(period: string): void {
    this.activeRange = period;

    this.http
      .post<any>('http://127.0.0.1:8000/history', { period: period })
      //.post<any>('/api/history', { period: period })
      .subscribe(
        (response) => {
          const dates = response.dates.map((d: string) => new Date(d));
          this.datesResumen = dates;
          const pointRadius = this.getPointRadius(dates);

          const data = response.dates.map((date: string, index: number) => ({
            x: new Date(date),
            y: response.values[index],
          }));

          this.xAxisMinResumen = dates[0];
          this.xAxisMaxResumen = dates[dates.length - 1];

          this.chartDataResumen = {
            datasets: [
              {
                label: 'Real',
                data: data,
                borderColor: '#00aa76',
                borderWidth: 2,
                pointRadius: pointRadius,
                pointHoverRadius: 8,
                tension: 0.1,
              },
            ],
          };

          this.updateXAxisLabels(dates);

          // Solo actualizar chartData si estamos en pestaña 'resumen'
          if (this.activeTab === 'resumen') {
            this.chartData = this.chartDataResumen;

            const xScale = this.chartOptions.scales!['x'] as any;

            if (dates.length === 1) {
              const singleDate = dates[0].getTime();
              const padding = 24 * 60 * 60 * 1000; // 1 día a cada lado
              xScale.min = singleDate - padding;
              xScale.max = singleDate + padding;
            } else {
              xScale.min = dates[0].getTime();
              xScale.max = dates[dates.length - 1].getTime();
            }

            xScale.bounds = 'ticks';
            xScale.offset = true;

            // Forzar actualización
            this.chartOptions = { ...this.chartOptions };
          }
        },
        (error) => {
          console.error('Error al obtener histórico, error', error);
          alert(
            'No se han podido obtener los datos históricos. Inténtelo de nuevo más tarde.'
          );
        }
      );
  }

  /**
   * Obtener predicciones y superponer con reales del rango
   */
  getPrediction(): void {
    if (!this.validateDates()) return;

    this.isLoading = true;

    this.http
      .post<any>('http://127.0.0.1:8000/normal_prediction', {
        //.post<any>('/api/normal_prediction', {
        start_date: this.startDate,
        end_date: this.endDate,
      })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          if (!response || !response.dates || response.dates.length === 0) {
            alert('No hay datos disponibles para las fechas seleccionadas.');
            this.isLoading = false;
            return;
          }

          const dates = response.dates.map((d: string) => new Date(d));
          this.datesNormal = dates;
          const pointRadius = this.getPointRadius(dates);

          const predictionData = response.dates.map(
            (date: string, index: number) => ({
              x: new Date(date),
              y: parseFloat(response.predictions[index].toFixed(2)),
            })
          );

          const realData = response.dates.map(
            (date: string, index: number) => ({
              x: new Date(date),
              y: response.real_values[index],
            })
          );

          this.xAxisMinNormal = dates[0];
          this.xAxisMaxNormal = dates[dates.length - 1];

          // Mostrar ambas líneas: reales y predicciones
          this.chartDataNormal = {
            datasets: [
              {
                label: 'Real',
                data: realData,
                borderColor: '#00aa76',
                borderWidth: 2,
                pointRadius: pointRadius,
                pointHoverRadius: 8,
                tension: 0.1,
              },
              {
                label: 'Predicción',
                data: predictionData,
                borderColor: 'red',
                fill: false,
                borderWidth: 2,
                pointRadius: pointRadius,
                pointHoverRadius: 8,
                tension: 0.1,
              },
            ],
          };

          this.updateXAxisLabels(dates);

          if (this.activeTab === 'normal') {
            this.chartData = this.chartDataNormal;

            const xScale = this.chartOptions.scales!['x'] as any;

            if (dates.length === 1) {
              const singleDate = dates[0].getTime();
              const padding = 24 * 60 * 60 * 1000;
              xScale.min = singleDate - padding;
              xScale.max = singleDate + padding;
            } else {
              xScale.min = dates[0].getTime();
              xScale.max = dates[dates.length - 1].getTime();
            }

            xScale.bounds = 'ticks';
            xScale.offset = true;

            // Forzar actualización
            this.chartOptions = { ...this.chartOptions };
          }
        },
        error: (error) => {
          console.error('Error al obtener predicciones', error);
          alert(
            'No se han podido obtener las predicciones. Inténtelo de nuevo más tarde.'
          );
          this.isLoading = false;
        },
        complete: () => {
          this.isLoading = false;
        },
      });
  }

  /**
   * Predicción recursiva
   */
  getRecursivePrediction(): void {
    if (!this.validateFutureDays()) return;

    this.isLoading = true;

    this.http
      .post<any>('http://127.0.0.1:8000/recursive_prediction', {
        //.post<any>('/api/recursive_prediction', {
        days: this.futureDays,
      })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          if (!response || !response.dates || response.dates.length === 0) {
            alert('No hay datos disponibles para los días solicitados.');
            this.isLoading = false;
            return;
          }

          const dates = response.dates.map((d: string) => new Date(d));
          this.datesRecursiva = dates;
          const pointRadius = this.getPointRadius(dates);

          const formattedData = response.dates.map(
            (date: string, index: number) => ({
              x: new Date(date),
              y: parseFloat(Number(response.predictions[index]).toFixed(2)),
            })
          );

          this.xAxisMinRecursiva = dates[0];
          this.xAxisMaxRecursiva = dates[dates.length - 1];

          // Mostrar predicción recursiva
          this.chartDataRecursiva = {
            datasets: [
              {
                label: 'Predicción',
                data: formattedData,
                borderColor: 'red',
                fill: false,
                pointRadius: pointRadius,
                pointHoverRadius: 8,
                borderWidth: 2,
                tension: 0.1,
              },
            ],
          };

          this.updateXAxisLabels(dates);

          if (
            this.activeTab === 'recursiva' &&
            this.xAxisMinRecursiva &&
            this.xAxisMaxRecursiva
          ) {
            this.chartData = this.chartDataRecursiva;

            const xScale = this.chartOptions.scales!['x'] as any;

            xScale.min = this.xAxisMinRecursiva;
            xScale.max = this.xAxisMaxRecursiva;

            xScale.bounds = 'data';

            // Forzar actualización
            this.chartOptions = { ...this.chartOptions };
          }
        },
        error: (error) => {
          console.error('Error al obtener predicciones', error);
          alert(
            'No se han podido obtener las predicciones. Inténtelo de nuevo más tarde.'
          );
          this.isLoading = false;
        },
        complete: () => {
          this.isLoading = false;
        },
      });
  }

  /**
   * Cambiar pestaña activa y mostrar gráfico correspondiente
   */
  setActiveTab(tabName: string): void {
    this.destroy$.next();
    this.destroy$ = new Subject<void>();
    this.isLoading = false;

    this.activeTab = tabName;

    let dates: Date[] = [];

    if (tabName === 'resumen' && this.datesResumen.length > 0) {
      this.chartData = this.chartDataResumen;
      dates = this.datesResumen;
      this.xAxisMinResumen = dates[0];
      this.xAxisMaxResumen = dates[dates.length - 1];
    } else if (tabName === 'normal' && this.datesNormal.length > 0) {
      this.chartData = this.chartDataNormal;
      dates = this.datesNormal;
      this.xAxisMinNormal = dates[0];
      this.xAxisMaxNormal = dates[dates.length - 1];
    } else if (tabName === 'recursiva' && this.datesRecursiva.length > 0) {
      this.chartData = this.chartDataRecursiva;
      dates = this.datesRecursiva;
      this.xAxisMinRecursiva = dates[0];
      this.xAxisMaxRecursiva = dates[dates.length - 1];
    } else {
      this.chartData = { datasets: [] };
      const xScale = this.chartOptions.scales!['x'] as any;
      xScale.min = undefined;
      xScale.max = undefined;
      this.chartOptions = { ...this.chartOptions };
      return;
    }

    const xScale = this.chartOptions.scales!['x'] as any;
    xScale.min = dates[0].getTime();
    xScale.max = dates[dates.length - 1].getTime();
    xScale.bounds = 'ticks';
    xScale.offset = true;

    this.updateXAxisLabels(dates);
    this.chartOptions = { ...this.chartOptions };
  }

  resetApp(): void {
    window.location.reload();
  }
}
