import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { provideHttpClient } from '@angular/common/http';
import { Chart, TimeSeriesScale, LinearScale, LineController, LineElement, PointElement, Tooltip, Legend } from 'chart.js';
import 'chartjs-adapter-date-fns';  // Adaptador para fechas

Chart.register(TimeSeriesScale, LinearScale, LineController, LineElement, PointElement, Tooltip, Legend);

bootstrapApplication(AppComponent, {
  providers: [provideHttpClient()]
});
