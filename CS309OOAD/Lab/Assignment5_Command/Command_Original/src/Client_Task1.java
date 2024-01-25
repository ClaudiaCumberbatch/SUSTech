import java.util.InputMismatchException;
import java.util.Scanner;

public class Client_Task1 {
    public static void main(String[] args) {
        AirConditioner roomAirConditioner = new AirConditioner("bedRoom");
        AirConditioner livingAirConditioner = new AirConditioner("livingRoom");
        Light roomLight = new Light("bedRoom");
        Light livingLight = new Light("livingRoom");
        Command[] commands = new Command[8];
        commands[0] = new AirConditionerOnCommand(roomAirConditioner);
        commands[1] = new AirConditionerOffCommand(roomAirConditioner);
        commands[2] = new AirConditionerOnCommand(livingAirConditioner);
        commands[3] = new AirConditionerOffCommand(livingAirConditioner);
        commands[4] = new LightOnCommand(roomLight);
        commands[5] = new LightOffCommand(roomLight);
        commands[6] = new LightOnCommand(livingLight);
        commands[7] = new LightOffCommand(livingLight);
        RemoteCommandQueue remoteCommandQueue = new RemoteCommandQueue();
        Scanner input = new Scanner(System.in);
        System.out.println("Please input operation number to add a command: 1-9," +
                "[1,3,5,7] is on command,[2,4,6,8] is off command, " +
                "9 is to show state. 10 is to execute command. terminate by 0:");
        int op = 0;
        do {
            try {
                op = input.nextInt();
                switch (op) {
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                    case 7:
                    case 8:
                        remoteCommandQueue.buttonPressed(commands[op - 1]);
                        break;
                    case 9:
                        showState(new AirConditioner[]{roomAirConditioner, livingAirConditioner}
                                , new Light[]{roomLight, livingLight});
                        break;
                    case 10:
                        remoteCommandQueue.commandExecute();
                        break;
                    case 11:
                        remoteCommandQueue.commandUndo();
                        break;
                }
            } catch (InputMismatchException e) {
                System.out.println("Exception:" + e);
                input.nextLine();
            }
        } while (op != 0);
        input.close();
    }

    public static void showState(AirConditioner[] airConditioners, Light[]
            lights) {
        for (AirConditioner a : airConditioners) {
            System.out.println(a);
        }
        for (Light l : lights) {
            System.out.println(l);
        }
    }
}
