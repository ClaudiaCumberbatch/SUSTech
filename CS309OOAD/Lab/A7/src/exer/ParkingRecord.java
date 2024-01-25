package exer;

import java.time.temporal.ChronoUnit;

public class ParkingRecord {
    private Car car;

    private double parkingHours;

    public ParkingRecord(TimeRange timeRange, Car car) {
        this.parkingHours = ChronoUnit.MINUTES.between(timeRange.arriveTime(), timeRange.departureTime()) / 60.0;
        this.car = car;
    }

    public String getCarNumber() {
        return car.getCarNumber();
    }

    public double getParkingHours() {
        return parkingHours;
    }

    public Car getCar() {
        return this.car;
    }

    @Override
    public String toString() {
        return String.format("%s -> parking %.1f hours", this.car.getCarNumber(),  this.parkingHours);
    }
}
